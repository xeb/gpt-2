#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]

import fire
import json
import numpy as np
import tensorflow as tf

import model, sample, encoder

import tflex

def clear_output(wait=False):
  import subprocess, platform
  if platform.system()=="Windows":
      subprocess.Popen("cls", shell=True).communicate()
  else:
      print("\033c", end="")

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    step=1,
    length=64,
    prompt="\n",
    clear=None,
    maxlen=-1,
    temperature=1,
    top_k=0,
    top_p=0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :step=1 : Number of tokens to generate at a time
    :length=64 : Window size; use 1024 for maximum size per sample
    :prompt="" : Prompt to start with. The default of "" prompts with an <|endoftext|> token.
    :clear="<|endoftext|>" : If this is encountered, clear the context window.
    :maxlen=-1 : if this many tokens are generated without
     encountering --clear, then print it and clear the context window.
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Length can't be largeer than n_ctx: %s" % hparams.n_ctx)
    if step > length:
        raise ValueError("Can't get samples longer than length: %s" % length)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=step,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tflex.Saver()
        ckpt = tflex.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
          try:
            with open(prompt) as f:
              raw_text = f.read()
          except:
            raw_text = prompt
          raw_text = raw_text.replace('\\n', '\n')
          raw_text = raw_text.replace('\\t', '\t')
          #print(repr(raw_text))
          context_tokens = enc.encode(raw_text) if len(raw_text) > 0 else [50256]
          while len(context_tokens) > length - step - 1:
            context_tokens = context_tokens[1:]
          prompt_tokens = context_tokens[:]
          first = True
          backlog = []
          backlog_count = 0
          context_text = ""
          context_count = 0
          while True:
            for tokens in generate_result(context_tokens=context_tokens, enc=enc, output=output, context=context, nsamples=1, batch_size=batch_size, sess=sess):
              if first:
                #clear_output(wait=True)
                sys.stdout.write(enc.decode(context_tokens))
                sys.stdout.flush()
                first = False
              backlog.extend(tokens)
              backlog_count += 1
              if is_ascii(enc.decode([backlog[-1]])) or backlog_count > 16:
                text = enc.decode(backlog)
                result = text
                if clear is not None:
                  result, *rest = text.split(clear)
                sys.stdout.write(result)
                sys.stdout.flush()
                context_text += text
                context_count += len(backlog)
                if maxlen > 0 and context_count > maxlen or clear is not None and clear in context_text:
                  context_text = ""
                  context_count = 0
                  context_tokens = []
                  first = True
                  tokens = prompt_tokens[:]
                backlog = []
                backlog_count = 0
              context_tokens.extend(tokens)
              while len(context_tokens) > length - step - 1:
                context_tokens = context_tokens[1:]

def generate_result(context_tokens, enc, output, context, nsamples=1, batch_size=1, sess=tf.get_default_session()):
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        for i in range(batch_size):
            yield out[i]

if __name__ == '__main__':
    fire.Fire(interact_model)

