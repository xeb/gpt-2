#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>
import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]

import argparse
import json
import numpy as np
import tensorflow as tf
import time
import tqdm
import math
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow

import model, sample, encoder
from load_dataset import load_dataset, Sampler, TextSampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients
from glob import glob
import re
import tflex
import tflex_sgdr

import pytz
from datetime import datetime, timezone

import threading
from collections import defaultdict

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00015, help='Learning rate for Adam')
parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='Minimum learning rate')
parser.add_argument('--learning_rate_cos', default=False, action='store_true', help='Use learn rate cosine annealing')
parser.add_argument('--learning_rate_warmup', type=int, default=100, help='Learning rate warmup for cosine annealing')
parser.add_argument('--learning_rate_period', type=int, default=100, help='Learning rate period for cosine annealing')
parser.add_argument('--learning_rate_initial_step', type=int, default=0, help='Learning rate initial step for cosine annealing')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd|ada>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=-1, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=-1, help='Write a checkpoint every N steps')
parser.add_argument('--save_time', metavar='N', type=float, default=15.0, help='Write a checkpoint every N minutes')
parser.add_argument('--max_to_keep', metavar='N', type=int, default=5, help='Only keep the last N checkpoints')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=1, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=80, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')

parser.add_argument('--init_tpu', default=False, action='store_true', help='Initialize TPU session.')

parser.add_argument('--fresh_model', default=False, action='store_true', help="Don't load model from disk; initialize model weights to random values")
parser.add_argument('--save_on_ctrlc', default=False, action='store_true', help='When execution is interrupted, should we save the model to disk?')
parser.add_argument('--debug_on_ctrlc', default=False, action='store_true', help='When execution is interrupted, attach a debugger (pdb.set_trace())')
parser.add_argument('--float16', default=False, action='store_true', help='Use float16 weights?')
parser.add_argument('--dtype', type=str, default='float32', help='dtype. <float32|float16|bfloat16>.')

parser.add_argument('--targets', type=str, default='', help='')

# 1.5B
#parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
#parser.add_argument('--n_embd', type=int, default=1600, help='For a fresh model, how large should n_embd be?')
#parser.add_argument('--n_head', type=int, default=25, help='For a fresh model, how large should n_head be?')
#parser.add_argument('--n_layer', type=int, default=48, help='For a fresh model, how large should n_layer be?')

# 345M
#parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
#parser.add_argument('--n_embd', type=int, default=1024, help='For a fresh model, how large should n_embd be?')
#parser.add_argument('--n_head', type=int, default=16, help='For a fresh model, how large should n_head be?')
#parser.add_argument('--n_layer', type=int, default=24, help='For a fresh model, how large should n_layer be?')

# 117M
#parser.add_argument('--n_ctx', type=int, default=1024, help='For a fresh model, how large should n_ctx be?')
#parser.add_argument('--n_embd', type=int, default=768, help='For a fresh model, how large should n_embd be?')
#parser.add_argument('--n_head', type=int, default=12, help='For a fresh model, how large should n_head be?')
#parser.add_argument('--n_layer', type=int, default=12, help='For a fresh model, how large should n_layer be?')

parser.add_argument('--n_ctx', type=int, default=-1, help='For a fresh model, how large should n_ctx be?')
parser.add_argument('--n_embd', type=int, default=-1, help='For a fresh model, how large should n_embd be?')
parser.add_argument('--n_head', type=int, default=-1, help='For a fresh model, how large should n_head be?')
parser.add_argument('--n_layer', type=int, default=-1, help='For a fresh model, how large should n_layer be?')

parser.add_argument('--sample_ctx', type=int, default=-1, help='Compute loss over N samples. Equal to n_ctx if set < 0.')

parser.add_argument('--truncate_weights', default=False, action='store_true', help="Try loading variables from snapshots, even if those variables' shapes do not match")

parser.add_argument('--debug_print_all_vars', default=False, action='store_true', help="Print all variables after running one training step")
parser.add_argument('--debug_print_trainable_vars', default=False, action='store_true', help="Print trainable variables after running one training step")

parser.add_argument('--allow_growth', default=False, action='store_true', help="Set config.gpu_options.allow_growth = True")
parser.add_argument('--allow_soft_placement', default=False, action='store_true', help="Set config.gpu_options.allow_soft_placement = True")
parser.add_argument('--disable_layout_optimizer', default=False, action='store_true', help="Set config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF")

parser.add_argument('--debug_before_training', default=False, action='store_true', help="Drop into debugger before starting the training loop")

parser.add_argument('--dropout', type=float, default=0.0, help="Dropout value. Disabled if set <= 0.0. For training on large datasets, 0.1 tends to be a good value.")

parser.add_argument('--seed', type=int, default=-1, help='Deterministic seed for dataset sampler. Disabled if set < 0')

parser.add_argument('--save_graph', default=False, action='store_true', help="Save TensorFlow graph to summary log (to see ops in tensorboard)")

parser.add_argument('--device', type=int, default=-1, help='device to use.')

PST = pytz.timezone('US/Pacific')

def timestamp(now=None, tz=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if tz is None:
        tz = PST
    return "{}".format(now.astimezone(tz).isoformat())

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

class TrainGPT2(object):
  def __init__(self, args, hparams, sampler, enc, scope='model', target='auto', timeout=120000, session=None):
    self.args = args
    self.hparams = hparams
    self.sampler = sampler
    self.target = target
    self.enc = enc
    if session is None:
      config = config_pb2.ConfigProto(operation_timeout_in_ms=timeout)
      self.timeout = timeout
      config.allow_soft_placement = False
      if args.allow_soft_placement:
          config.allow_soft_placement = True
      if args.allow_growth:
          config.gpu_options.allow_growth = True
      if args.disable_layout_optimizer:
          config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
      session = tflex.Session(target=target, config=config, init_tpu=args.init_tpu)

    with session.as_default():
      cores = session.list_devices()[2:]
      core = cores[args.device].name if len(cores) > 0 and args.device >= 0 else None
      #with tf.device(core):
      if True:
        #context = tf.placeholder(tf.int32, [args.batch_size, None])
        context = tf.Variable(tf.zeros(shape=[args.batch_size, args.sample_ctx], dtype=tf.int32), dtype=tf.int32, name="context", trainable=False)
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in, scope=scope)
        loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

      all_vars = [v for v in tf.trainable_variables() if v.name.startswith(scope + '/')]
      train_vars = [v for v in all_vars if '/h' in v.name or '/ln_f' in v.name] if args.only_train_transformer_layers else all_vars

      parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
      print("This model is using %d parameters (%.2fM)" % (parameter_count, parameter_count/(1024.0*1024.0)))

      with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
        global_step = tflex.get_variable('global_step') or tf.get_variable('global_step', shape=(), dtype=tf.int32, trainable=False)
        current_step = args.learning_rate_initial_step
        lr = tflex.get_variable('learn_rate') or tf.get_variable('learn_rate', shape=(), dtype=tf.float32, trainable=False)

        if args.optimizer == 'adam':
          opt = tf.train.AdamOptimizer(learning_rate=lr)
        elif args.optimizer == 'sgd':
          opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif args.optimizer == 'ada':
          import tensor2tensor.utils.optimize
          from tensor2tensor.utils import hparam
          import tensor2tensor.models.research
          from tensor2tensor.utils import registry
          ada_hparams = registry.hparams('afx_mimic_adam')
          ada_hparams.optimizer_adafactor_beta1 = 0.0
          ada_hparams.optimizer_adafactor_factored = True
          opt = tensor2tensor.utils.optimize.adafactor(learning_rate=lr, hparams=ada_hparams)
        else:
          exit('Bad optimizer:', args.optimizer)

        opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)
        summary_perp = tf.summary.scalar('perplexity', tf.math.exp(loss))

      summary_lr = tf.summary.scalar('learning_rate', lr)
      summaries = tf.summary.merge([summary_lr, summary_loss, summary_perp])
      self.summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name + "_" + self.target))
      self.summaries = summaries
      self.loss = loss
      self.context = context
      self.output = output
      self.opt = opt
      self.all_vars = all_vars
      self.train_vars = train_vars
      self.opt_grads = opt_grads
      self.opt_apply = opt_apply
      self.sess = session
      self.lr = lr
      self.counter = 1
      self.current_step = current_step
      self.global_step = global_step
      self.saver = tflex.Saver(
            var_list=all_vars,
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=2,
            reshape=args.truncate_weights)
      self.init = tf.global_variables_initializer()
      self.avg_loss = (0.0, 0.0)
      self.avg_perp = (0.0, 0.0)
    self.start_time = time.time()
    self.prev_time = self.start_time
    

  def sample_batch(self):
    args = self.args
    return [self.sampler.sample(args.sample_ctx) for _ in range(args.batch_size)]
  
  def elapsed(self):
    return time.time() - self.start_time

  def say(self, msg):
    print('{stamp} [{counter} | {time:2.4f}] {msg}'.format(counter=self.counter, time=self.elapsed(), msg=msg, stamp=timestamp()))

  def update_lr(self):
    self.lr.load(self.args.learning_rate, session=self.sess)
    return self.lr.eval(session=self.sess)

  def fit(self):
    with self.sess.as_default():
      if self.init is not None:
        self.say('Initializing...')
        self.sess.run(self.init, options=config_pb2.RunOptions(timeout_in_ms=self.timeout))
        args = self.args
        if not args.fresh_model:
          self.say('Restoring...')
          saver = self.saver
          if args.restore_from == 'latest':
            ckpt = tflex.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
              # Get fresh GPT weights if new run.
              ckpt = tflex.latest_checkpoint(os.path.join('models', args.model_name))
          elif args.restore_from == 'fresh':
            ckpt = tflex.latest_checkpoint(os.path.join('models', args.model_name))
          else:
            ckpt = tflex.latest_checkpoint(args.restore_from)
          print('Loading snapshot %s...' % ckpt)
          t0 = time.time()
          if not args.fresh_model:
            saver.restore(self.sess, ckpt)
          t1 = time.time()
          print('Loaded in %f seconds' % (t1 - t0))
        #global_step.load(current_step, session=session)
        #lr.load(args.learning_rate, session=session)
        self.init = None
      v_rate = self.update_lr()
      self.say('Generating batch...')
      batch = self.sample_batch()
      print(repr(self.enc.decode(batch[0]))[0:150] + '...')
      self.say('Loading context...')
      self.context.load(batch, session=self.sess)
      self.say('Running opt_apply...')
      (_, v_loss, v_summary) = self.sess.run((self.opt_apply, self.loss, self.summaries), options=config_pb2.RunOptions(timeout_in_ms=self.timeout))
      self.avg_loss = (self.avg_loss[0] * 0.99 + v_loss,
                       self.avg_loss[1] * 0.99 + 1.0)
      v_perp = math.exp(v_loss)
      self.avg_perp = (self.avg_perp[0] * 0.99 + v_perp,
                       self.avg_perp[1] * 0.99 + 1.0)
      now = time.time()
      print('{stamp} [{counter} | {time:2.4f} | {delta:2.2f}s | {ops:2.6f}tokens/s] loss={loss:2.4f} avg={avg:2.4f} perp={perp:2.4f} avg_perp={avg_perp:2.4f} rate={rate:0.7f} step={step}'
          .format(
              stamp=timestamp(),
              counter=self.counter,
              time=now - self.start_time,
              delta=now - self.prev_time,
              ops=self.args.sample_ctx * self.args.batch_size / (now - self.prev_time),
              rate=v_rate,
              loss=v_loss,
              avg=self.avg_loss[0] / self.avg_loss[1],
              perp=v_perp,
              avg_perp=self.avg_perp[0] / self.avg_perp[1],
              step=self.current_step,
              ))
      self.prev_time = now
      self.summary_log.add_summary(v_summary, self.counter)
      self.summary_log.flush()
      self.counter += 1
      self.current_step += 1
      self.global_step.load(self.current_step, session=self.sess)

    return v_loss

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    hparams.res_dropout = args.dropout
    hparams.attn_dropout = args.dropout
    epsilon = -1e10
    if args.dtype == 'float32':
        hparams.dtype = tf.float32
    elif args.dtype == 'float16':
        hparams.dtype = tf.float16
        epsilon = -65500
    elif args.dtype == 'bfloat16':
        hparams.dtype = tf.bfloat16
        epsilon = -65500
    else:
        print('Unknown dtype', args.dtype)
    if args.float16:
        hparams.dtype = tf.bfloat16
        epsilon = -65500

    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if args.n_ctx >= 0:
        hparams.n_ctx=args.n_ctx
    if args.n_embd >= 0:
        hparams.n_embd=args.n_embd
    if args.n_head >= 0:
        hparams.n_head=args.n_head
    if args.n_layer >= 0:
        hparams.n_layer=args.n_layer

    if args.sample_length < 0:
        args.sample_length = hparams.n_ctx - 1
    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)
    if args.sample_ctx < 0:
      args.sample_ctx = hparams.n_ctx

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    def make_sampler(dataset, enc, seed, combine):
      if os.path.isdir(dataset) or dataset.endswith('.npz'):
        chunks = load_dataset(enc, dataset, combine)
        data_sampler = Sampler(chunks, seed=seed)
        print('dataset has', data_sampler.total_size, 'tokens', len(chunks), 'chunks')
      else:
        data_sampler = TextSampler(dataset, enc, seed=seed, use_locking=True)
      return data_sampler

    print('Loading dataset...')
    seed = None if args.seed < 0 else args.seed
    data_sampler = make_sampler(dataset=args.dataset, enc=enc, seed=seed, combine=args.combine)

    local = threading.local()

    targets = [x.strip() for x in args.targets.split(',') if len(x.strip()) > 0]
    if len(targets) <= 0:
      targets.append('auto')
    trainers = [TrainGPT2(args=args, hparams=hparams, sampler=data_sampler, enc=enc, target=target) for target in targets]
    i = 0
    while True:
      tflex.check_commands()
      if tflex.should_quit():
        break
      i += 1
      threads = []
      for trainer in trainers:
        def thunk(trainer):
          trainer.fit()
        thread = threading.Thread(target=thunk, args=(trainer,))
        thread.start()
        threads.append(thread)
      for thread in threads:
        thread.join()
      print('All done', i)
      if len(trainers) > 1 and i % 10 == 0:
        def sync():
          print('Fetching...')
          accum = {}
          accumcount = defaultdict(int)
          lock = threading.Lock()
          threads = []
          for trainer in trainers:
            def thunk(trainer):
              for variables, values in trainer.saver.fetch(trainer.sess):
                try:
                  lock.acquire()
                  for variable, value in zip(variables, values):
                    if variable.name in accum:
                      accum[variable.name] = accum[variable.name] + value
                    else:
                      accum[variable.name] = value
                    accumcount[variable.name] += 1
                finally:
                  lock.release()
            thread = threading.Thread(target=thunk, args=(trainer,))
            thread.start()
            threads.append(thread)
          for thread in threads:
            thread.join()
          print('Synchronizing...')
          threads = []
          for trainer in trainers:
            def thunk(trainer):
              for variables in trainer.saver.variables(trainer.sess):
                values = []
                for v in variables:
                  assert(v.name in accum)
                  value = accum[v.name]
                  n = accumcount[v.name]
                  assert(n > 0)
                  values.append(value / n)
                trainer.saver.assign(trainer.sess, variables, values)
            thread = threading.Thread(target=thunk, args=(trainer,))
            thread.start()
            threads.append(thread)
          for thread in threads:
            thread.join()
          print('Synchronized.')
        thread = threading.Thread(target=sync, args=())
        thread.start()

if __name__ == '__main__':
    main()

