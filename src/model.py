import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import math
import tflex
from collections import OrderedDict
from tensorflow.python.framework import ops as tf_ops

def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        res_dropout=0.1,
        attn_dropout=0.1,
        dtype=tf.float32
    )

 #_cores = None
 #
 #def get_core(i, session=None):
 #  global _cores
 #  if session is None:
 #    session = tf.get_default_session()
 #  if _cores is None:
 #    _cores = session.list_devices()[2:]
 #  n = len(_cores)
 #  if n <= 0:
 #    return None
 #  result = _cores[i % n].name
 #  if 'GPT2_VERBOSE' in os.environ:
 #    print(result)
 #  return result

def get_cores(session=None):
  if session is None:
    session = tf.get_default_session()
  cores = session.list_devices()[2:2+8]
  cores = cores[::-1]
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
        g = get_variable('g') or tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=dtype))
        b = get_variable('b') or tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=dtype))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, u, v = shape_list(x)
    m = u * v
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = get_variable('w') or tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dtype))
        b = get_variable('b') or tf.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=dtype))
        lhs = tf.reshape(x, [-1, nx])
        rhs = tf.reshape(w, [-1, nf])
        if False: # noticeable slowdown https://i.imgur.com/95VAycJ.png
          lhs_n = tf.split(lhs, 8, axis=1)
          rhs_n = tf.split(rhs, 8, axis=0)
          ops = []
          for i in range(8):
            with tf.device(get_core(i)):
              ops.append(tf.matmul(lhs_n[i], rhs_n[i]))
          W = tf.reduce_sum(ops, axis=0)
        else:
          W = tf.matmul(lhs, rhs)
        lhs1 = W+b
        rhs1 = start+[nf]
        c = tf.reshape(lhs1, rhs1)
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams, batch_size, seq_length):
    assert x.shape.ndims == 2  # Should be [batch*sequence, features]
    assert n_state % hparams.n_head == 0
    *start, hidden_size = shape_list(x)
    num_attention_heads = hparams.n_head
    assert(hidden_size % num_attention_heads == 0)
    size_per_head = hidden_size // num_attention_heads

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
        x = split_states(x, hparams.n_head)
        return tf.transpose(x, [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        x = merge_states(x)
        x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
        return x

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
        q, k, v = map(split_heads, tf.split(c, 3, axis=-1))
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

def block(x, scope, *, past, hparams, attn, **attn_kws):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1', hparams=hparams), 'attn', nx, past=past, hparams=hparams, **attn_kws)
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


def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE, checkpoint=False):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = get_variable('wpe') or tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype))
        wte = get_variable('wte') or tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        ## We keep the representation as a 2D tensor to avoid re-shaping it back and
        ## forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        ## the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        ## help the optimizer.
        batch_size, seq_length, hidden_size = shape_list(h)
        h = tf.reshape(h, [batch_size * seq_length, hidden_size])

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        #every = int(math.sqrt(hparams.n_layer))
        every = 1
        #tf.add_to_collection('checkpoints', h)
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams,
                attn=attn, batch_size=batch, seq_length=sequence)
            #if layer == 10:
            if checkpoint and layer % every == 0:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        if hparams.dtype != tf.float32:
          logits = tf.cast(logits, tf.float32)
        results['logits'] = logits
        return results

from tensorflow.python.ops import gradients
import memory_saving_gradients

class Shard(object):
  pass

def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def bfloat16context(hparams):
  if hparams.dtype == tf.bfloat16:
    return tf.contrib.tpu.bfloat16_scope()
  else:
    return nullcontext()

def shape_to_list(shape):
    """Convert a Tensorflow shape to a list of ints."""
    return [dim.value for dim in shape]

def is_tf_expression(x):
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))

def absolute_name_scope(scope):
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def absolute_variable_scope(scope, **kwargs):
    """Forcefully enter the specified variable scope, ignoring any surrounding scopes."""
    return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)


def all_sum_plain(g, colocate=True, *args, **kws):
  r = []
  for i in range(len(g)):
    if colocate:
      with tf_ops.colocate_with(g[i]):
        r.append(tf.add_n(g))
    else:
      r.append(tf.add_n(g))
  return r

try:
    # TensorFlow 1.13
    from tensorflow.python.ops import nccl_ops
except:
    # Older TensorFlow versions
    import tensorflow.contrib.nccl as nccl_ops

def all_sum_gpu(g, *args, **kws):
  return nccl_ops.all_sum(g, *args, **kws)

from tensorflow.python.tpu.ops import tpu_ops

#def all_sum_tpu(g, *args, **kws):
#  g = tpu_ops.cross_replica_sum(g, *args, **kws)
#  return [g[i] for i in range(shape_list(g)[0])]

def all_sum_tpu(g, colocate=True, *args, **kws):
  #import pdb
  #pdb.set_trace()
  #r = tf.reduce_sum(g)
  #r = tf.reduce_sum(tf.stack(g), axis=0, keepdims=True)
  #r = tpu_ops.cross_replica_sum(g, *args, **kws)
  #r = [r[i] for i in range(shape_list(r)[0])]
  return all_sum_plain(g, colocate=colocate, *args, **kws)

def all_sum(cores, g, colocate=True, *args, **kws):
  if any(['TPU' in x for x in cores]):
    return all_sum_tpu(g, colocate=colocate, *args, **kws)
  elif any(['GPU' in x for x in cores]):
    return all_sum_gpu(g, *args, **kws)
  else:
    return all_sum_cpu(g, *args, **kws)

def init_uninitialized_vars(target_vars = None, run = lambda ops: [False] * len(ops)):
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    #assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    return [var.initializer for var in init_vars]

def shard(batch_size, hparams, learning_rate=0.0001, optimizer='sgd', noise=0.0, only_train_transformer_layers=False, colocate_gradients_with_ops=False, colocate_sum=False, use_memory_saving_gradients=False, ungate_gradients=False, global_step=None, graph=None, scope='model', skip_cores=4, max_cores=4, length=None, sample_ctx=None, encoder=None, temperature=1, top_k=0, top_p=0.0, devices=None, *args, **kws):
    if graph is None:
        graph = tf.get_default_graph()
    if length is None:
        length = hparams.n_ctx
    if sample_ctx is None:
        sample_ctx = length
    results = {}
    results['shards'] = {}
    #results = {}
    #results['present'] = []
    #results['logits'] = []
    _dev_grads = results['grads'] = OrderedDict()
    with graph.as_default():
      #batch_size, *rest = shape_list(X)
      if devices is None:
        devices = get_cores()
      else:
        devices = devices[2:2+8]
      cores = [x.name for x in devices[skip_cores:]]
      num_cores = len(cores)
      if max_cores < 0:
        num_core = 1
      elif max_cores is not None:
        if num_cores > max_cores:
          num_cores = max_cores
      if num_cores > batch_size:
        num_cores = batch_size
      assert(num_cores > 0)
      cores = cores[:num_cores]
      #if num_cores <= 0:
      #  return model(hparams, X, scope=scope, *args, **kws)
      print('Sharding across %d cores' % len(cores))
      assert(batch_size % num_cores == 0)
      #contexts = tf.split(X, num_cores, axis=0)
      def make_shard(i):
        with graph.as_default():
          return make_shard_1(i)
      def make_shard_1(i):
        core = cores[i] if i >= 0 else None
        prefix = ('core%04d' % i) if i >= 0 else None
        #context = contexts[i]
        #context = tf.placeholder(tf.int32, [batch_size // num_cores, None])
        #context_in = randomize(context, hparams, noise)
        with tf.device(core), bfloat16context(hparams), tf.variable_scope(prefix, reuse=tf.AUTO_REUSE):
          context = tf.Variable(tf.zeros(shape=[batch_size // num_cores, sample_ctx], name="context", dtype=tf.int32), dtype=tf.int32, shape=[batch_size // num_cores, sample_ctx], trainable=False)
          #context_set = tf.placeholder(tf.int32, [batch_size // num_cores, None])
          #feed_op = tf.assign(context, context_set)
          context_in = randomize(context, hparams, noise)
          output = model(hparams=hparams, X=context_in, scope=scope, checkpoint=use_memory_saving_gradients, *args, **kws)
          #if hparams.dtype == tf.bfloat16:
          #  output['logits'] = tf.cast(output['logits'], tf.float32)
          infer = None
          if encoder:
            infer = sample.sample_sequence(
                hparams=hparams, length=length,
                start_token=encoder.encoder['<|endoftext|>'],
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )[:, 1:]

          loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=context[:, 1:], logits=output['logits'][:, :-1])
          loss = tf.reduce_mean(loss_batch)
          #if hparams.dtype != tf.float32:
          #    loss = tf.cast(loss, tf.float32)

        path = scope
        if prefix is not None:
          path = prefix + '/' + path
        global_vars = [v for v in tf.global_variables() if v.name.startswith(path + '/')]
        all_vars = [v for v in tf.trainable_variables() if v.name.startswith(path + '/')]
        def should_train_variable(v):
          if only_train_transformer_layers:
            if '/h' not in v.name and '/ln_f' not in v.name:
              return False
            #for i in range(1):
            #  if ('/h%01d/' % i) in v.name:
            #    return False
            #  if ('/h%02d/' % i) in v.name:
            #    return False
          print(v)
          return True
        train_vars = [v for v in all_vars if should_train_variable(v)]

        parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
        print("Shard %d is using %d parameters (%.2fM) (scope='%s/')" % (i, parameter_count, parameter_count/(1024.0*1024.0), path))

        with tf.device(core), bfloat16context(hparams), tf.variable_scope(prefix, reuse=tf.AUTO_REUSE):
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
          elif optimizer == None:
            pass
          else:
              exit('Bad optimizer:', optimizer)
          _dev_grads[core] = []
          r = Shard()
          r.prefix = prefix
          r.scope = scope
          r.context = context
          r.context_in = context_in
          #r.context_set = context_set
          #r.feed_op = feed_op
          r.device = core
          r.output = output
          r.infer = infer
          if optimizer is not None:
            #opt_apply = opt.minimize(loss, var_list=train_vars, global_step=global_step, colocate_gradients_with_ops=colocate_gradients_with_ops)
            gate_gradients=None
            if ungate_gradients:
              gate_gradients=tf.train.Optimizer.GATE_NONE
            if use_memory_saving_gradients:
              #grads = memory_saving_gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, checkpoints='memory')
              #grads = memory_saving_gradients.gradients_memory if i == 0 else memory_saving_gradients.gradients_speed
              #grads = memory_saving_gradients.gradients_speed if i == 0 else memory_saving_gradients.gradients_speed
              grads = memory_saving_gradients.gradients
              grads = grads(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
            else:
              grads = gradients.gradients(loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
            grads = list(zip(grads, train_vars))
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]  # replace disconnected gradients with zeros
            _dev_grads[core].append(grads)
            #opt_apply = opt.apply_gradients(grads, global_step=global_step)
            #fit = tf.tuple([loss], control_inputs=[opt_apply])
            r.loss_batch = loss_batch
            r.loss = loss
            r.opt = opt
            #r.opt_apply = opt_apply
            #r.fit = fit
          r.global_vars = global_vars
          r.all_vars = all_vars
          r.train_vars = train_vars
          r.global_vars = global_vars
          r.parameter_count = parameter_count
          results['shards'][i] = r
        #results['present'].append(r['present'])
        #results['logits'].append(r['logits'])
      #for thread in tflex.parallelize([i for i in range(num_cores)], make_shard):
      #  thread.join()
      for i in range(num_cores):
        make_shard(i)
      #import pdb
      #pdb.set_trace()
      results['shards'] = [v for i, v in sorted(list(results['shards'].items()))]

      total_grads = sum(len(grads) for grads in _dev_grads.values())
      assert len(cores) >= 1 and total_grads >= 1
      use_loss_scaling = False

      # Cast gradients to FP32 and calculate partial sum within each device.
      dev_grads = OrderedDict()  # device => [(grad, var), ...]

      for dev_idx, dev in enumerate(cores):
          with tf.name_scope("ProcessGrads%d" % dev_idx), tf.device(dev):
              sums = []

              for gv in zip(*_dev_grads[dev]):
                  assert all(v is gv[0][1] for g, v in gv)
                  g = [tf.cast(g, tf.float32) for g, v in gv]
                  g = g[0] if len(g) == 1 else tf.add_n(g)
                  sums.append((g, gv[0][1]))

              dev_grads[dev] = sums

      trainable_vars = results['shards'][0].train_vars
      _grad_shapes = [shape_to_list(var.shape) for var in trainable_vars]

      # Sum gradients across cores.
      if len(cores) > 1:
          with tf.name_scope("SumAcrossGPUs"), tf.device(None):
              for var_idx, grad_shape in enumerate(_grad_shapes):
                  g = [dev_grads[dev][var_idx][0] for dev in cores]

                  if np.prod(grad_shape):  # nccl does not support zero-sized tensors
                    g = all_sum(cores, g, colocate=colocate_sum)

                  for dev, gg in zip(cores, g):
                      dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

      # Apply updates separately on each device.
      ops = []
      for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
          with tf.name_scope("ApplyGrads%d" % dev_idx), tf.device(dev):
              # Scale gradients as needed.
              if use_loss_scaling or total_grads > 1:
                  with tf.name_scope("Scale"):
                      coef = tf.constant(np.float32(1.0 / total_grads), name="coef")
                      #coef = undo_loss_scaling(coef)
                      grads = [(g * coef, v) for g, v in grads]

              # Check for overflows.
              #with tf.name_scope("CheckOverflow"):
              #    grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

              # Update weights and adjust loss scaling.
              with tf.name_scope("UpdateWeights"):
                  # pylint: disable=cell-var-from-loop
                  #opt = _dev_opt[dev]
                  opt = results['shards'][dev_idx].opt
                  #ls_var = get_loss_scaling_var(dev)
                  #ls_var = None

                  #if not use_loss_scaling:
                  #    ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                  #else:
                  #    ops.append(tf.cond(grad_ok,
                  #                       lambda: tf.group(tf.assign_add(ls_var, loss_scaling_inc), opt.apply_gradients(grads)),
                  #                       lambda: tf.group(tf.assign_sub(ls_var, loss_scaling_dec))))
                  ops.append(opt.apply_gradients(grads))
      opt_apply = tf.group(*ops, name="TrainingOp")

      #present = tf.concat(results['present'], axis=0)
      #logits = tf.concat(results['logits'], axis=0)
      #import pdb
      #pdb.set_trace()
      #results['present'] = present
      #results['logits'] = logits
      inputs = [x.context for x in results['shards']]
      results['inputs'] = inputs
      def get_feed_dict(batch, session=None, options=None):
        session = session or tf.get_default_session()
        r = {}
        j = batch_size // num_cores
        parts = tflex.tuples(j, batch)
        assert(batch_size % num_cores == 0)
        shards = results['shards']
        def load(i, verbose=False):
          context = inputs[i]
          tokens = parts[i]
          #r[context] = tokens
          shard = shards[i]
          with tf.device(shard.device):
            if verbose:
              print('Loading context', i)
            #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
            tflex.load(shard.context, tokens, session=session, timeout_in_ms=15000)
            #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
            if verbose:
              print('Loaded context', i)
        for thread in tflex.parallelize([i for i in range(len(inputs))], load):
          thread.join()
        #for i, context in enumerate(inputs):
        #  tokens = parts[i]
        #  #r[context] = tokens
        #  shard = shards[i]
        #  with tf.device(shard.device):
        #    print('Loading context')
        #    #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
        #    tflex.load(shard.context, tokens, session=session, timeout_in_ms=15000)
        #    #session.run(shard.feed_op, feed_dict={shard.context_set: tokens}, options=options)
        #    print('Loaded')
        return r
      results['feed'] = get_feed_dict
      shards = results['shards']
      opt_losses = [x.loss for x in shards]
      opt_loss = tf.reduce_mean(opt_losses)
      #the_vars = [x.global_vars for x in shards]
      the_vars = [x.train_vars for x in shards]
      gather_ops = []
      variable_count = len(the_vars[0])
      shard_count = len(the_vars)
      #opt_apply = tf.tuple([x.fit for x in shards])
      #opt_apply = tf.group([x.opt_apply for x in shards])
      #opt_apply = tf.group([shards[i].opt_apply for i in range(1,shard_count)])
      #opt_apply = tf.group([x.opt_apply for x in shards[1:]])
      #dev_grads = OrderedDict() # device => [(val, var), ...]
      #devices = [x.device for x in shards]
      #for dev_idx, dev in enumerate(devices):
      #  with tf.name_scope("GatherWeights%d" % dev_idx), tf.device(dev):
      #    #sums = []
      #    #for gv in zip(*self._dev_grads[dev]):
      #    #   assert all(v is gv[0][1] for g, v in gv)
      #    #   g = [tf.cast(g, tf.float32) for g, v in gv]
      #    #   g = g[0] if len(g) == 1 else tf.add_n(g)
      #    #   sums.append((g, gv[0][1]))
      #    #dev_grads[dev] = sums
      #    vs = []
      #    for variables in zip(*the_vars):
      #    for variables in the_vars[dev_idx]:
      #        g = [tf.cast(g, tf.float32) for g in variables]
      #        g = g[0] if len(g) == 1 else tf.add_n(g)
      #        sums.append((g, gv[0][1]))


      ##    dev_grads[dev] = sums
      pivot = 0
      #for j in range(variable_count):
      #  with tf.device(shards[pivot].device):
      #    gather_ops.append(tf.assign(the_vars[pivot][j], tf.reduce_mean([the_vars[i][j] for i in range(shard_count)], axis=0), use_locking=True))
      #with tf.control_dependencies(gather_ops):
      #  opt_gather = tf.no_op()
      #broadcast_ops = []
      #for i in range(shard_count):
      #  if i != pivot:
      #    with tf.device(shards[i].device):
      #      for j in range(variable_count):
      #        broadcast_ops.append(tf.assign(the_vars[i][j], the_vars[pivot][j], use_locking=True))
      #  ##op1 = tf.group([tf.assign(the_vars[i][j], op0) for i in range(0,shard_count)])
      #  ##op0 = tf.reduce_sum([(the_vars[i][j] - the_vars[0][j]) for i in range(shard_count)], axis=0) / (shard_count/2) + the_vars[0][j]
      #  ##op0 = tf.reduce_mean([(the_vars[i][j] - the_vars[0][j]) for i in range(shard_count)], axis=0) * interp_rate + the_vars[0][j]
      #  ##op0 = tf.reduce_sum([(the_vars[i][j] - the_vars[0][j]) for i in range(1,shard_count)], axis=0) / (shard_count - 1) * interp_rate + the_vars[0][j]
      #  ##op0 = tf.reduce_mean([(the_vars[i][j] - the_vars[0][j]) for i in range(1,shard_count)], axis=0) * interp_rate + the_vars[0][j]
      #  #op1 = tf.group([tf.assign(the_vars[i][j], op0) for i in range(0,shard_count)])
      #  #gather_ops.append(op1)
      #with tf.control_dependencies(broadcast_ops):
      #  opt_broadcast = tf.no_op()
      ##opt_gather = tf.group(gather_ops)
      ##opt_gather = tf.group(gather_ops)
      opt_gather = tf.no_op()
      opt_broadcast = tf.no_op()
      all_vars = [x.all_vars for x in shards]
      reset_ops = []
      reset_var = {}
      variable_count = len(all_vars[pivot])
      shard_count = len(all_vars)
      for j in range(variable_count):
        reset_op = tf.group([tf.assign(all_vars[i][j], all_vars[pivot][j]) for i in range(shard_count) if i != pivot])
        reset_var[all_vars[pivot][j].name] = reset_op
        reset_var[all_vars[pivot][j]] = reset_op
        reset_ops.append(reset_op)
      #opt_reset = tf.group(reset_ops)
      #def init():
      #  init_op = tf.variables_initializer(shards[0].global_vars)
      #  with tf.control_dependencies([init_op]):
      #    return tf.group(reset_ops)
      #opt_init = init()
      #with tf.control_dependencies(init_uninitialized_vars(shards[0].all_vars)):
      #  ops = [reset_var[v] for v in shards[0].all_vars]
      #init_ops = init_uninitialized_vars(shards[0].all_vars)
      #opt_init = tf.group(init_ops, name="Initialize")
      opt_init = tf.global_variables_initializer()
      #opt_train = tf.tuple([x.loss for x in shards], control_inputs=[x.opt_apply for x in shards])
      #opt_train = tf.tuple([x.loss for x in shards], control_inputs=[x.opt_apply for x in shards[1:]])
      opt_train = tf.tuple([x.loss for x in shards], control_inputs=[opt_apply])
      the = tflex.Namespace()
      results['the'] = the
      the.opt_losses = opt_losses
      the.opt_loss = opt_loss
      the.opt_apply = opt_apply
      the.opt_gather = opt_gather
      the.opt_broadcast = opt_broadcast
      the.opt_train = opt_train
      the.reset_ops = reset_ops
      the.reset_var = reset_var
      the.opt_init = opt_init
      the.vars = the_vars
      the.all_vars = all_vars
      return results
