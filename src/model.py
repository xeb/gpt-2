import os
import numpy as np
import tensorflow as tf
from optimizers import *
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import gradients
import memory_saving_gradients
import math

def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        res_dropout=0.0,
        attn_dropout=0.0,
        dtype=tf.float32
    )

def get_cores(session=None):
  if session is None:
    session = tf.get_default_session()
  cores = session.list_devices()[2:2+8]
  return cores

def get_core(i, session=None):
  cores = get_cores(session=session)
  if len(cores) > 0:
    return cores[i % len(cores)].name

def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
            return x

def get_or_init_variable(name, *args, **kws):
  v = get_variable(name)
  if v is not None:
    return v
  #use_resource = kws.pop('use_resource') if 'use_resource' in kws else True
  use_resource = kws.pop('use_resource') if 'use_resource' in kws else False
  v = tf.get_variable(name, use_resource=use_resource, *args, **kws)
  return v

def constant_initializer(*args, **kws):
  #with tf.device(None):
  #  return tf.constant_initializer(*args, **kws)
  return tf.constant_initializer(*args, **kws)

def random_normal_initializer(*args, **kws):
  #with tf.device(None):
  #  return tf.random_normal_initializer(*args, **kws)
  #x = kws.pop('stddev')
  #return constant_initializer(x, *args, **kws)
  return tf.random_normal_initializer(*args, **kws)

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        n_state = x.shape[-1].value
        g = get_or_init_variable('g', [n_state], initializer=constant_initializer(1, dtype=dtype))
        b = get_or_init_variable('b', [n_state], initializer=constant_initializer(0, dtype=dtype))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = get_or_init_variable('w', [1, nx, nf], initializer=random_normal_initializer(stddev=w_init_stdev, dtype=dtype))
        b = get_or_init_variable('b', [nf], initializer=constant_initializer(0, dtype=dtype))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(65500 if w.dtype == tf.float16 else 1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state, hparams=hparams))
        h2 = conv1d(h, 'c_proj', nx, hparams=hparams)
        h2 = dropout(h2, hparams.res_dropout)
        return h2

def dropout(x, pdrop=0.1, train=True):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

def block(x, scope, *, past, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1', hparams=hparams), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2', hparams=hparams), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = get_or_init_variable('wpe', [hparams.n_ctx, hparams.n_embd],
            initializer=random_normal_initializer(stddev=0.01, dtype=dtype))
        wte = get_or_init_variable('wte', [hparams.n_vocab, hparams.n_embd],
            initializer=random_normal_initializer(stddev=0.02, dtype=dtype))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        every = int(math.sqrt(hparams.n_layer))
        #every = 1
        #tf.add_to_collection('checkpoints', h)
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            #if layer == 10:
            if layer % every == 0:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)
        #tf.add_to_collection('checkpoints', h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        #tf.add_to_collection('checkpoints', h_flat)
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results

def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

class Namespace(object):
  pass

def shard(batch_size, hparams, learning_rate, optimizer='sgd', noise=0.0, only_train_transformer_layers=False, colocate_gradients_with_ops=False, use_memory_saving_gradients=False, global_step=None, session=None, scope='model', max_cores=8, *args, **kws):
    if session is None:
        session = tf.get_default_session()
    results = {}
    results['shards'] = []
    #results = {}
    #results['present'] = []
    #results['logits'] = []
    with session.as_default():
      #batch_size, *rest = shape_list(X)
      cores = get_cores()
      num_cores = len(cores)
      if max_cores is not None:
        if num_cores > max_cores:
          num_cores = max_cores
      if num_cores > batch_size:
        num_cores = batch_size
      assert(num_cores > 0)
      #if num_cores <= 0:
      #  return model(hparams, X, scope=scope, *args, **kws)
      print('Sharding across %d cores' % len(cores))
      assert(batch_size % num_cores == 0)
      #contexts = tf.split(X, num_cores, axis=0)
      for i in range(num_cores):
        core = cores[i].name
        use_scope = scope if i <= 0 else ('core%03d_%s' % (i, scope))
        #context = contexts[i]
        context = tf.placeholder(tf.int32, [batch_size // num_cores, None])
        context_in = randomize(context, hparams, noise)
        with tf.device(core):
          if hparams.dtype == tf.bfloat16:
            with tf.contrib.tpu.bfloat16_scope():
              output = model(hparams=hparams, X=context_in, scope=use_scope, *args, **kws)
            output['logits'] = tf.cast(output['logits'], tf.float32)
          else:
            output = model(hparams=hparams, X=context_in, scope=use_scope, *args, **kws)

        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1])
        loss = tf.reduce_mean(loss_batch)
        if hparams.dtype != tf.float32:
            loss = tf.cast(loss, tf.float32)

        global_vars = [v for v in tf.global_variables() if v.name.startswith(use_scope + '/')]
        all_vars = [v for v in tf.trainable_variables() if v.name.startswith(use_scope + '/')]
        train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars

        parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
        print("Shard %d is using %d parameters (%.2fM) (scope='%s/')" % (i, parameter_count, parameter_count/(1024.0*1024.0), use_scope))

        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer == 'ada':
          params = {}
          params["decay_type"] = "adam"
          #params["beta1"] = 0.9
          params["beta1"] = 0.0
          params["beta2"] = 0.999
          lr = learning_rate
          if params["decay_type"] == "adam":
              decay_rate = adafactor_decay_rate_adam(params["beta2"])
          elif params["decay_type"] == "pow":
              decay_rate = adafactor_decay_rate_pow(params["decay_exponent"])
          else:
              raise ValueError("unknown optimizer_adafactor_decay_type")

          if not "weight_decay" in params.keys():
              opt = AdafactorOptimizer(
                  learning_rate=lr,
                  decay_rate=decay_rate,
                  beta1=params["beta1"],
                  name="Adafactor")

          else:
              AdafactorWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(AdafactorOptimizer)

              opt = AdafactorWOptimizer(
                  weight_decay=params["weight_decay"] * lr,
                  learning_rate=lr,
                  decay_rate=decay_rate,
                  beta1=params["beta1"],
                  name="AdafactorW")
        elif optimizer == 'ada':
            import tensor2tensor.utils.optimize
            from tensor2tensor.utils import hparam
            import tensor2tensor.models.research
            from tensor2tensor.utils import registry
            ada_hparams = registry.hparams('afx_mimic_adam')
            ada_hparams.optimizer_adafactor_beta1 = 0.0
            ada_hparams.optimizer_adafactor_factored = True
            opt = tensor2tensor.utils.optimize.adafactor(learning_rate=learning_rate, hparams=ada_hparams)
        else:
            exit('Bad optimizer:', optimizer)
        #opt_apply = opt.minimize(loss, var_list=train_vars, global_step=global_step, colocate_gradients_with_ops=colocate_gradients_with_ops)
        if use_memory_saving_gradients:
          #opt_grads = memory_saving_gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, checkpoints='memory')
          opt_grads = memory_saving_gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops)
        else:
          opt_grads = gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads, global_step=global_step)
        fit = tf.tuple([loss], control_inputs=[opt_apply])
        r = Namespace()
        r.scope = use_scope
        r.context = context
        r.context_in = context_in
        r.output = output
        r.loss_batch = loss_batch
        r.loss = loss
        r.all_vars = all_vars
        r.train_vars = train_vars
        r.global_vars = global_vars
        r.parameter_count = parameter_count
        r.opt = opt
        r.opt_apply = opt_apply
        r.fit = fit
        results['shards'].append(r)
        #results['present'].append(r['present'])
        #results['logits'].append(r['logits'])
      #present = tf.concat(results['present'], axis=0)
      #logits = tf.concat(results['logits'], axis=0)
      #import pdb
      #pdb.set_trace()
      #results['present'] = present
      #results['logits'] = logits
      inputs = [x.context for x in results['shards']]
      results['inputs'] = inputs
      def get_feed_dict(batch):
        r = {}
        j = batch_size // num_cores
        assert(batch_size % num_cores == 0)
        for i, context in enumerate(inputs):
          r[context] = batch[i:i+j]
        return r
      results['feed'] = get_feed_dict
      return results
