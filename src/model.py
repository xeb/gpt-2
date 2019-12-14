import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import math

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

import os

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
    *start, m = shape_list(x)
    #start = [np.prod(start)]
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
        #c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        x0 = tf.reshape(x, [-1, nx])
        w0 = tf.reshape(w, [-1, nf])
        z = tf.matmul(x0, w0)
        c = tf.reshape(z+b, start+[nf])
        #import pdb
        #pdb.set_trace()
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams, **kws):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    #import pdb
    #pdb.set_trace()
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        #import pdb
        #pdb.set_trace()
        x = split_states(x, hparams.n_head)
        return tf.transpose(x, [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        #import pdb
        #pdb.set_trace()
        x = merge_states(x)
        return x

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch*heads, sequence, features]
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
        x = tf.split(c, 3, axis=-1)
        #import pdb
        #pdb.set_trace()
        q, k, v = map(split_heads, x)
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        #import pdb
        #pdb.set_trace()
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present

def _attention_projection_and_transpose(x_flat, batch_size, seq_length, num_attention_heads, size_per_head,
                                        name, initializer_range=0.02, project=False):
    """
    :param x_flat: [batch_size*seq_length, width]
    :return: A fixed up tensor of size [batch_size, num_attention_heads, seq_length, size_per_head]
    """
    #batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)
    *batch_size_seq_length, dim = shape_list(x_flat)
    batch_size_seq_length = np.prod(batch_size_seq_length)

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        #import pdb
        #pdb.set_trace()
        x = split_states(x, num_attention_heads)
        return tf.transpose(x, [0, 2, 1, 3])

    # Had to remove this bc of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    #projected = tf.layers.dense( #    x_flat,
    #    num_attention_heads * size_per_head,
    #    name=name,
    #    kernel_initializer=create_initializer(initializer_range))

    if project:
      n_state = num_attention_heads * size_per_head
      projected = conv1d(x_flat, 'c_attn_'+name, n_state)
    else:
      projected = x_flat

    projected = tf.reshape(
        projected, [batch_size, seq_length, num_attention_heads, size_per_head])
    #import pdb
    #pdb.set_trace()
    output_tensor = tf.transpose(projected, [0, 2, 1, 3])
    #output_tensor = split_heads(projected)
    return output_tensor

def mask_attn_weights(w, mask):
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    _, _, nd, ns = shape_list(w)
    #b = attention_mask(nd, ns, dtype=w.dtype)
    b = mask
    b = tf.reshape(b, [1, 1, nd, ns])
    w = w*b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype)*(1-b)
    return w

def attention_layer(x_flat, attention_mask, batch_size, seq_length, size_per_head=512, num_attention_heads=1, *,
                    cache=None,
                    initializer_range=0.02, hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1, do_cache=False):
    """

    :param x_flat: Tensor input, should be [batch_size*seq_length, dim]
    :param attention_mask: Attention mask to use of size [seq_length, seq_length+cached_length]
    :param size_per_head: dim = size_per_head * num_attention_heads
    :param num_attention_heads:  dim = size_per_head * num_attention_heads
    :param cache: Optionally some past (cached) things of size
                [batch, 2, heads, sequence, features], where 2 is [k, v]
    :param do_cache: True if we should return cache
    :return: A new tensor of shape [batch_size, seq_length, dim]
    as well as a new cache "cached_keys_and_values" that will be of size
                                   [batch_size, 2, num_attention_heads, seq_length, dim]
    """
    #*batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)
    *batch_size_seq_length, dim = shape_list(x_flat)
    batch_size_seq_length = np.prod(batch_size_seq_length)

    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        #import pdb
        #pdb.set_trace()
        x = merge_states(x)
        return x

    # Had to remove this because of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    # if do_cache and past is not None:
    #     Shape will be (batch_size, 2, num_attention_heads, past_seq_length, dim)
    #     past_shape = get_shape_list(past, 5)
    #     desired_shape = (batch_size, 2, num_attention_heads, seq_length, dim)
    #     if tuple(past_shape) != desired_shape:
    #         raise ValueError(f"The shape of the cache is {past_shape} but we want {desired_shape}")

    #project=True
    project=False

    if not project:
      n_state = num_attention_heads * size_per_head * 3
      x = conv1d(x_flat, 'c_attn', n_state)
      q_flat, k_flat, v_flat = tf.split(x, 3, axis=-1)
    else:
      q_flat, k_flat, v_flat = [x_flat, x_flat, x_flat]

    # [ batch_size, num_attention_heads, seq_length, size_per_head]
    query = _attention_projection_and_transpose(q_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='q',
                                                project=project,
                                                initializer_range=initializer_range)
    key = _attention_projection_and_transpose(k_flat, batch_size=batch_size, seq_length=seq_length,
                                              num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                              name='k',
                                              project=project,
                                              initializer_range=initializer_range)

    value = _attention_projection_and_transpose(v_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='v',
                                                project=project,
                                                initializer_range=initializer_range)

    # Add to cache
    cached_keys_and_values = tf.stack([key, value], axis=1) if do_cache else None

    # Things that were relevant from the cache
    if cache is not None:
        pk, pv = tf.unstack(cache, axis=1)
        key = tf.concat([pk, key], axis=-2)
        value = tf.concat([pv, value], axis=-2)

    # Multiply [batch_size, num_attention_heads, seq_length, size_per_head] with
    #          [batch_size, num_attention_heads, size_per_head, seq_length+cached_length] ->
    #          [batch_size, num_attention_heads, seq_length, seq_length+cached_length]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    attention_scores = mask_attn_weights(attention_scores, attention_mask)

    #attention_scores = mask_attention_for_ltr(attention_scores, attention_mask)
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    # NOPENOPENOPENOPE
    # attention_probs = factoreddropout(attention_probs, attention_probs_dropout_prob)

    # Multiply [batch_size, num_attention_heads, seq_length, seq_length+cached_length] with
    #          [batch_size, num_attention_heads, seq_length+cached_length, size_per_head] ->
    #          [batch_size, num_attention_heads, seq_length, size_per_head] ->
    context_layer = tf.matmul(attention_probs, value)

    # `context_layer` = [batch_size, seq_length, num_attention_heads, size_per_head]
    #context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    context_layer = merge_heads(context_layer)
    #import pdb
    #pdb.set_trace()
    #context_layer = tf.reshape(context_layer, [batch_size, seq_length, num_attention_heads * size_per_head])
    #context_layer = tf.reshape(context_layer, [batch_size*seq_length, num_attention_heads * size_per_head])

    #context_layer_projected = tf.layers.dense(
    #    context_layer,
    #    num_attention_heads * size_per_head,
    #    kernel_initializer=create_initializer(initializer_range),
    #    name='c_proj'
    #)
    n_state = num_attention_heads * size_per_head
    context_layer_projected = conv1d(context_layer, 'c_proj', n_state)
    context_layer_projected = dropout(context_layer_projected, hidden_dropout_prob)

    return context_layer_projected, cached_keys_and_values


def attn2(hidden_state, scope, nx, past, hparams, mask, batch_size, seq_length):
    #batch_size, seq_length, hidden_size = shape_list(hidden_state)
    hidden_size = hparams.n_embd
    num_attention_heads = hparams.n_head
    hidden_dropout_prob = hparams.res_dropout
    attention_probs_dropout_prob = hparams.attn_dropout
    with tf.variable_scope(scope):
        # [batch_size * seq_length, hidden_size]
        attention_output, present = attention_layer(
            hidden_state,
            mask,
            batch_size=batch_size,
            seq_length=seq_length,
            size_per_head=hidden_size // num_attention_heads,
            num_attention_heads=num_attention_heads,
            #initializer_range=config.initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            #do_cache=do_cache,
            #cache=layer_cache,
        )
    return attention_output, present

def mlp(x, scope, n_state, *, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        x1 = conv1d(x, 'c_fc', n_state, hparams=hparams)
        h = gelu(x1)
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
        ln_1 = norm(x, 'ln_1', hparams=hparams)
        a, present = attn(ln_1, 'attn', nx, past=past, hparams=hparams, **attn_kws)
        x = x + a
        ln_2 = norm(x, 'ln_2', hparams=hparams)
        m = mlp(ln_2, 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(embedding_table, input_ids, past_length):
    batch_size = tf.shape(input_ids)[0]
    nsteps = tf.shape(input_ids)[1]
    return tf.gather(embedding_table, expand_tile(past_length + tf.range(nsteps), batch_size))

def gather_for(full_position_embeddings, input_ids):
    #return tf.gather(full_position_embeddings, input_ids) #h = tf.gather(wte, X)
    return tf.nn.embedding_lookup(full_position_embeddings, input_ids)

def embed(input_ids,
          vocab_size,
          embedding_size,
          max_position_embeddings,
          position_offset,
          initializer_range_wpe=0.01,
          initializer_range_wte=0.02,
          ):
    wpe = get_variable('wpe') or tf.get_variable('wpe',
        shape=[max_position_embeddings, embedding_size], #[hparams.n_ctx, hparams.n_embd],
        initializer=tf.random_normal_initializer(stddev=initializer_range_wpe))

    wte = get_variable('wte') or tf.get_variable('wte',
        shape=[vocab_size, embedding_size], #[hparams.n_vocab, hparams.n_embd],
        initializer=tf.random_normal_initializer(stddev=initializer_range_wte))

    #past_length = 0 if past is None else tf.shape(past)[-2]
    h = gather_for(wte, input_ids)
    h = h + positions_for(wpe, input_ids, position_offset)
    return wpe, wte, h

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

from utils import get_shape_list, get_attention_mask#, gelu, layer_norm, dropout

def embed2(input_ids,
          vocab_size,
          embedding_size,
          position_offset=0,
          initializer_range=0.02,
          max_position_embeddings=512,
          use_one_hot_embeddings=True):
    """reur and position embeddings
    :param input_ids: int Tensor of shape [batch_size, seq_length].
    :param vocab_size: number of words in vocab
    :param embedding_size: dimensionality of the embedding
    :param position_offset: aka number of cached tokens.
    :param initializer_range: float. Range of the weight initialization.
    :param max_position_embeddings: int. Maximum sequence length.
    :param use_one_hot_embeddings: probably want this to be true
    :return: [batch_size, seq_length, embedding_size] embedded tensor
    """
    (batch_size, seq_length) = get_shape_list(input_ids, expected_rank=2)

    full_position_embeddings = tf.get_variable(
        name='wpe',
        shape=[max_position_embeddings, embedding_size],
        initializer=create_initializer(initializer_range),
    )

    embedding_table = tf.get_variable(
        name='wte',
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
    )

    #assert_op = tf.assert_less_equal(tf.reduce_max(input_ids), vocab_size - 1)
    assert_op = tf.no_op()
    with tf.control_dependencies([assert_op]):
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output_flat = tf.matmul(one_hot_input_ids, embedding_table)
        else:
            output_flat = tf.nn.embedding_lookup(embedding_table, input_ids)

        embedded_input = tf.reshape(output_flat, [batch_size, seq_length, embedding_size])

    #assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    assert_op = tf.no_op()

    with tf.control_dependencies([assert_op]):
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        if position_offset == 0:
            embedded_input += tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])[None]
        else:
            # Tensorflow is too stupid to allow slicing
            flat_pos_ids = (tf.range(seq_length, dtype=tf.int32) + position_offset)
            one_hot_pos_ids = tf.one_hot(flat_pos_ids, depth=max_position_embeddings)

            # [seq_length, full_position_embeddings], [full_position_embeddings, dim]
            seq_embeds = tf.matmul(one_hot_pos_ids, full_position_embeddings)
            embedded_input += seq_embeds[None]

            # embedded_input += tf.slice(full_position_embeddings[position_offset:], [0, 0], [seq_length, -1])[None]

    #return layer_norm(embedded_input, name='ln_f'), embedding_table
    return full_position_embeddings, embedding_table, embedded_input

def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        past_length = 0 if past is None else tf.shape(past)[-2]
        batch, sequence = shape_list(X)
        #wpe, wte, h = embed(
        #    input_ids=X,
        #    position_offset=past_length,
        #    max_position_embeddings=hparams.n_ctx,
        #    vocab_size=hparams.n_vocab,
        #    embedding_size=hparams.n_embd)
        wpe, wte, h = embed2(
            input_ids=X,
            position_offset=past_length,
            max_position_embeddings=hparams.n_ctx,
            vocab_size=hparams.n_vocab,
            embedding_size=hparams.n_embd)
        #import pdb
        #pdb.set_trace()

        #wpe = get_variable('wpe') or tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
        #                     initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype))
        #wte = get_variable('wte') or tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
        #                     initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype))
        #h = gather_for(wte, input_ids)
        #h = h + positions_for(wpe, input_ids, past_length)

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        ## We keep the representation as a 2D tensor to avoid re-shaping it back and
        ## forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        ## the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        ## help the optimizer.
        #batch_size, seq_length, hidden_size = shape_list(h)
        import pdb
        pdb.set_trace()
        batch_size, seq_length, hidden_size = shape_list(h)
        #h = tf.reshape(h, [batch_size * seq_length, hidden_size])

        #mask = get_attention_mask(self.seq_length, self.seq_length + self.cache_length, dtype=embeddings.dtype)
        mask = get_attention_mask(sequence, sequence,# + self.cache_length,
            dtype=hparams.dtype)

        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams, attn=attn,
                mask=mask, batch_size=batch_size, seq_length=seq_length)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
