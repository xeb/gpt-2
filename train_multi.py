#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>
import os
import random
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
from tensorflow.python.ops import gradients
from tensorflow.python.framework.errors_impl import InvalidArgumentError, AbortedError, DeadlineExceededError

import model, sample, encoder
from load_dataset import load_dataset, Sampler, TextSampler, TokenSampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients
from glob import glob
import re
import tflex
import tflex_sgdr
import tflex_optimizers

import pytz
from datetime import datetime, timezone

import threading
from collections import defaultdict

# We allocate hundreds of threads. By default the OS might give ~2MB
# to each thread. In practice 128K seems to work fine.
threading.stack_size(32768 * 4)

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
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer. <adam|adamw|sgd|ada>.')
parser.add_argument('--weight_decay', metavar='WD', type=float, default=1e-4, help='Weight decay for AdamW/AdaW')
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
parser.add_argument('--dtype', type=str, default='float32', help='dtype. <float32|float16|bfloat16>.')

parser.add_argument('--targets', type=str, default='', help='')
parser.add_argument('--max_cores', type=int, default=4)
parser.add_argument('--skip_cores', type=int, default=4)

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
parser.add_argument('--colocate_gradients', default=False, action='store_true')
parser.add_argument('--colocate_sum', default=False, action='store_true')
parser.add_argument('--ungate_gradients', default=False, action='store_true', help="Use gate_gradients=tf.train.Optimizer.GATE_NONE")
parser.add_argument('--no_report_tensor_allocations_upon_oom', default=True, action='store_false')

parser.add_argument('--debug_before_training', default=False, action='store_true', help="Drop into debugger before starting the training loop")

parser.add_argument('--dropout', type=float, default=0.0, help="Dropout value. Disabled if set <= 0.0. For training on large datasets, 0.1 tends to be a good value.")

parser.add_argument('--seed', type=int, default=-1, help='Deterministic seed for dataset sampler. Disabled if set < 0')

parser.add_argument('--save_graph', default=False, action='store_true', help="Save TensorFlow graph to summary log (to see ops in tensorboard)")

parser.add_argument('--device', type=int, default=-1, help='device to use.')
parser.add_argument('--no_averaging', default=False, action='store_true')
parser.add_argument('--coreless', default=False, action='store_true')

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

class TrainCounter(object):
  def __init__(self, value=0):
    self.value = value
    self.lock = threading.Lock()

  def incr(self, n=1):
    try:
      self.lock.acquire()
      self.value += n
      return self.value
    finally:
      self.lock.release()

tflex.pinned_sessions = []
tflex.session_timeout_in_ms = 1200000
tflex.eval_lightweight_timeout = 20000
tflex.load_lightweight_timeout = 20000
#tflex.initialize_timeout = 60000
tflex.initialize_timeout = 20*60000
tflex.context_load_timeout = 20000
tflex.ensure_on_init = True
tflex.release_trainer_sema = True
tflex.tpu_init_timeout = 10000
tflex.summary_log_timeout = 20000
tflex.use_global_data_sampler = True
tflex.shuffle_cycles = True

def eval_lightweight(variable, session, timeout_in_ms=None):
  if timeout_in_ms is None:
    timeout_in_ms = tflex.eval_lightweight_timeout
  return tflex.eval(variable, session=session, timeout_in_ms=tflex.eval_lightweight_timeout)

def load_lightweight(variable, value, session, timeout_in_ms=None):
  if timeout_in_ms is None:
    timeout_in_ms = tflex.load_lightweight_timeout
  return tflex.load(variable, value, session=session, timeout_in_ms=timeout_in_ms)

tflex._cores = None

def get_cores(session=None):
  if session is None:
    session = tf.get_default_session()
  if tflex._cores is None:
    tflex._cores = session.list_devices()[2:2+8]
  return tflex._cores

tflex.get_cores = get_cores

def get_core(i, session=None):
  cores = tflex.get_cores(session=session)
  n = len(cores)
  if n <= 0:
    return None
  result = cores[i % n].name
  if 'GPT2_VERBOSE' in os.environ:
    print(result)
  return result

tflex.get_core = get_core

class TrainGPT2(object):
    
  #def aborted(self):
  #  try:
  #    self.sess.list_devices()
  #    return False
  #  except InvalidArgumentError:
  #    return True
  #  except AbortedError:
  #    return True
  #  except DeadlineExceededError:
  #    return True

  def sample_batch(self):
    return tflex.trainer_sample_batch(self)
  
  def elapsed(self):
    return tflex.trainer_elapsed(self)

  def say(self, msg):
    return tflex.trainer_say(self, msg)

  def update_lr(self, step=None, rate=None):
    return tflex.trainer_update_lr(self, step=step, rate=rate)

  def ensure(self):
    return tflex.trainer_ensure(self)

  def fit(self, *args, **kws):
    return tflex.trainer_fit(self, *args, **kws)

  def flush(self, *args, **kws):
    return tflex.trainer_flush(self, *args, **kws)

  def variables(self, index):
    return tflex.trainer_variables(self, index)

  @property
  def slices(self):
    return tflex.trainer_slices(self)

def trainer_fork(existing, target):
    self = TrainGPT2()
    for k, v in existing.__dict__.items():
      setattr(self, k, v)
    session = tflex.Session(target=target, graph=existing.sess.graph, config=existing.sess.config)
    if self.args.init_tpu:
      print('Initializing TPU...', session.target)
      config = config_pb2.ConfigProto(operation_timeout_in_ms=tflex.tpu_init_timeout)
      with tf.Session(target=target, graph=tf.Graph(), config=config) as sess:
        with sess.graph.as_default():
          sess.run(tf.contrib.tpu.initialize_system(), options=config_pb2.RunOptions(timeout_in_ms=tflex.tpu_init_timeout))
    self.summary_log = tflex.trainer_open_summary_log(run_name=self.args.run_name, target=target)
    self.sess = session
    self.init = self.init_op
    self.thread = threading.Thread(target=tflex.trainer_toplevel, args=(self,))
    self.lock = threading.RLock()
    self.pending_writes = []
    self.counter = self.current_step.value
    tflex.pinned_sessions.append([target, session]) # prevent GC'ing sessions, because the destructor seems to freeze.
    return self

tflex.trainer_fork = trainer_fork

def generate_nonce(length=6):
    """Generate pseudorandom number."""
    return ''.join([str(random.randint(0, 9)) for i in range(length)])

tflex.generate_nonce = generate_nonce

# return current UTC timestamp.
def utc():
    from datetime import datetime
    d = datetime.utcnow()
    import calendar
    return calendar.timegm(d.utctimetuple())

tflex.utc = utc

def trainer_open_summary_log(run_name, target, suffix=None):
  if suffix is None:
    suffix = str(utc())
  run_name = run_name + "_" + target + "_" + suffix
  run_name = run_name.replace('/', '_').replace(':', '_').replace('.', '_')
  return tf.summary.FileWriter(os.path.join(CHECKPOINT_DIR, run_name))

tflex.trainer_open_summary_log = trainer_open_summary_log

def trainer_create(args, hparams, enc, scope='model', target='auto', timeout=tflex.session_timeout_in_ms, session=None, counter=None):
    self = TrainGPT2()
    core = args.device
    if '::' in target:
      target, core = target.split('::')
      core = int(core)
      print(target, 'core', core)
    self.core = core
    self.fresh = True
    self.dead = False
    self.args = args
    self.hparams = hparams
    self.enc = enc
    if session is None:
      config = config_pb2.ConfigProto(operation_timeout_in_ms=timeout)
      self.timeout = timeout
      config.allow_soft_placement = False
      if args.allow_growth:
          config.gpu_options.allow_growth = True
      if args.allow_soft_placement:
          config.allow_soft_placement = True
      if args.disable_layout_optimizer:
          config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
      options = config_pb2.RunOptions(report_tensor_allocations_upon_oom=(not args.no_report_tensor_allocations_upon_oom))
      self.options = options
      session = tflex.Session(target=target, config=config)
      tflex.pinned_sessions.append([target, session]) # prevent GC'ing sessions, because the destructor seems to freeze.

    config = config_pb2.ConfigProto(operation_timeout_in_ms=tflex.tpu_init_timeout)
    with tf.Session(target=target, graph=tf.Graph(), config=config) as sess:
      with sess.graph.as_default():
        if args.init_tpu:
          print('Initializing TPU...', session.target)
          sess.run(tf.contrib.tpu.initialize_system(), options=config_pb2.RunOptions(timeout_in_ms=tflex.tpu_init_timeout))
        devices = sess.list_devices()
    #devices = tflex.get_cores(session=session)
    device = None
    if self.core >= 0:
      device = tflex.get_core(self.core, session=session) # not quite right; punt for now.
    self.device = device
    with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
      global_step = tflex.get_variable('global_step') or tf.get_variable('global_step', shape=(), dtype=tf.int32, trainable=False)
      current_step = counter
      #load_lightweight(global_step, current_step.value, session=session)
      if args.learning_rate_cos:
          lr = tflex_sgdr.sgdr_decay_with_warmup(args.learning_rate, global_step,
              warmup_steps=args.learning_rate_warmup, initial_period_steps=args.learning_rate_period, learning_rate_min=args.learning_rate_min)
      else:
          lr = tflex.get_variable('learn_rate') or tf.get_variable('learn_rate', shape=(), dtype=tf.float32, trainable=False)
          #load_lightweight(lr,args.learning_rate, session=session)
      wd = tflex.get_variable('weight_decay') or tf.get_variable('weight_decay', shape=(), dtype=tf.float32, trainable=False)
    with tf.device(device), tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
      ##context = tf.placeholder(tf.int32, [args.batch_size, None])
      #context = tf.Variable(tf.zeros(shape=[args.batch_size, args.sample_ctx], dtype=tf.int32), dtype=tf.int32, name="context", trainable=False)
      #context_in = randomize(context, hparams, args.noise)
      #output = model.model(hparams=hparams, X=context_in, scope=scope, checkpoint=args.memory_saving_gradients)
      #loss = tf.reduce_mean(
      #  tf.nn.sparse_softmax_cross_entropy_with_logits(
      #    labels=context[:, 1:], logits=output['logits'][:, :-1]))
      #if hparams.dtype == tf.bfloat16:
      #  loss = tf.cast(loss, tf.float32)
      output = model.shard(batch_size=args.batch_size, hparams=hparams, noise=args.noise, learning_rate=lr, optimizer=args.optimizer, only_train_transformer_layers=args.only_train_transformer_layers, colocate_gradients_with_ops=args.colocate_gradients, colocate_sum=args.colocate_sum, use_memory_saving_gradients=args.memory_saving_gradients, ungate_gradients=args.ungate_gradients,max_cores=-1 if args.coreless else args.max_cores, skip_cores=args.skip_cores, sample_ctx=args.sample_ctx, devices=devices)
      #use_locking=False
      #if args.optimizer == 'adam':
      #  opt = tf.train.AdamOptimizer(learning_rate=lr, use_locking=use_locking)
      #elif args.optimizer == 'adamw':
      #  opt = tflex_optimizers.AdamWOptimizer(learning_rate=lr, use_locking=use_locking, weight_decay=wd)
      #elif args.optimizer == 'sgd':
      #  opt = tf.train.GradientDescentOptimizer(learning_rate=lr, use_locking=use_locking)
      #elif args.optimizer == 'ada':
      #  import tensor2tensor.utils.optimize
      #  from tensor2tensor.utils import hparam
      #  import tensor2tensor.models.research
      #  from tensor2tensor.utils import registry
      #  ada_hparams = registry.hparams('afx_mimic_adam')
      #  ada_hparams.optimizer_adafactor_beta1 = 0.0
      #  ada_hparams.optimizer_adafactor_factored = True
      #  opt = tensor2tensor.utils.optimize.adafactor(learning_rate=lr, hparams=ada_hparams)
      #elif args.optimizer == 'adaw':
      #  opt = tflex_optimizers.AdafactorWOptimizer(learning_rate=lr, use_locking=use_locking, weight_decay=wd)
      #else:
      #  exit('Bad optimizer:', args.optimizer)
      #all_vars = [v for v in tf.trainable_variables() if v.name.startswith(scope + '/')]
      #train_vars = [v for v in all_vars if '/h' in v.name or '/ln_f' in v.name] if args.only_train_transformer_layers else all_vars
      #parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
      #print("This model is using %d parameters (%.2fM)" % (parameter_count, parameter_count/(1024.0*1024.0)))
      shards = output['shards']
      the = output['the']
      opt_loss = the.opt_loss
      #opt_apply = the.opt_apply
      #opt_gather = the.opt_gather
      #opt_train = the.opt_train
      #if args.memory_saving_gradients:
      #  opt_grads = memory_saving_gradients.gradients(loss, train_vars, colocate_gradients_with_ops=args.colocate_gradients)
      #else:
      #  opt_grads = gradients.gradients(loss, train_vars, colocate_gradients_with_device=args.colocate_gradients)
      #opt_grads = list(zip(opt_grads, train_vars))
      #opt_apply = opt.apply_gradients(opt_grads)
      summary_loss = tf.summary.scalar('loss', opt_loss)
      #summary_loss = tf.summary.scalar('loss', loss)
      summary_perp = tf.summary.scalar('perplexity', tf.math.exp(opt_loss))
      #global_vars = [v for v in tf.global_variables() if v.name.startswith(scope + '/')]
      global_vars = shards[0].global_vars
      train_vars = shards[0].train_vars
      all_vars = shards[0].all_vars
      fetch_global_vars = [list(tflex.split_by_params(shard.global_vars)) for shard in shards]
      fetch_train_vars = [list(tflex.split_by_params(shard.train_vars)) for shard in shards]
      #fetch_vars = list(tflex.split_by_params(all_vars))
      summary_lr = tf.summary.scalar('learning_rate', lr)
      #summary_wd = tf.summary.scalar('weight_decay', wd)
      #summaries = tf.summary.merge([summary_lr, summary_wd, summary_loss, summary_perp])
      summaries = tf.summary.merge([summary_lr, summary_loss, summary_perp])
      self.summaries = summaries
      self.summary_log = tflex.trainer_open_summary_log(run_name=args.run_name, target=target)
      #self.loss = loss
      #self.context = context
      self.output = output
      #self.opt = opt
      self.all_vars = all_vars
      self.train_vars = train_vars
      self.global_vars = global_vars
      self.fetch_global_vars = fetch_global_vars
      self.fetch_train_vars = fetch_train_vars
      self.fetch_vars = self.fetch_train_vars[0] if args.optimizer in ['adam', 'adamw'] else self.fetch_global_vars[0]
      #self.opt_grads = opt_grads
      #self.opt_apply = opt_apply
      self.sess = session
      self.lr = lr
      self.wd = wd
      self.counter = current_step.value
      self.stopped = False
      self.paused = False
      self.current_step = current_step
      self.global_step = global_step
      self.saver = tflex.Saver(
            var_list=all_vars,
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=2,
            reshape=args.truncate_weights)
      #self.init_op = tf.global_variables_initializer()
      self.init_op = output['the'].opt_init
      self.init = self.init_op
      self.avg_loss = [0.0, 0.0]
      self.avg_perp = [0.0, 0.0]
    self.start_time = time.time()
    self.prev_time = self.start_time
    self.thread = threading.Thread(target=tflex.trainer_toplevel, args=(self,))
    self.lock = threading.RLock()
    self.pending_writes = []
    return self

tflex.trainer_create = trainer_create

tflex.sample_lock = threading.Lock()
tflex.sample_step = 25
tflex.sample_ahead = 200

def trainer_sample_batch(self, count=None, length=None):
  if not hasattr(self, "samples"):
    self.samples = []
  if count is None:
    count = self.args.batch_size
  if length is None:
    length = self.args.sample_ctx
  size = self.hparams.n_ctx
  if len(self.samples) < count:
    with tflex.sample_lock:
      if len(self.samples) < count:
        self.say('Generating %d samples of %d tokens...' % (tflex.sample_ahead, size))
        for i in tqdm.tqdm(range(0, tflex.sample_ahead, tflex.sample_step)):
          for j in range(tflex.sample_step):
            if not hasattr(tflex, 'data_sampler') or tflex.data_sampler is None:
              print('Loading dataset...')
              seed = None if tflex.args.seed < 0 else tflex.args.seed
              tflex.data_sampler = tflex.make_sampler(dataset=tflex.args.dataset, enc=tflex.enc, seed=seed, combine=tflex.args.combine)
            tokens = tflex.data_sampler.sample(size)
            #print(repr(tokens), repr(size), len(tokens))
            if len(tokens) >= size:
              self.samples.append(tokens)
  return [self.samples.pop()[0:length] for _ in range(count)]

tflex.trainer_sample_batch = trainer_sample_batch

def trainer_elapsed(self):
  return self.prev_time - self.start_time

tflex.trainer_elapsed = trainer_elapsed

def trainer_say(self, msg):
  print('{stamp} {target:16s}::{core} [{counter} | {time:2.4f}] {msg}'.format(stamp=timestamp(), target=self.sess.target[-16:], core=self.core, counter=self.counter, time=self.elapsed(), msg=msg))

tflex.trainer_say = trainer_say

def trainer_variables(self, index, variables=None):
  if variables is None:
    variables = self.fetch_vars
  else:
    variables = list(tflex.split_by_params(variables))
  return variables[index % len(variables)]

tflex.trainer_variables = trainer_variables

def trainer_slices(self, variables=None):
  if variables is None:
    variables = self.fetch_vars
  else:
    variables = list(tflex.split_by_params(variables))
  return len(variables)

tflex.trainer_slices = trainer_slices

def trainer_update_lr(self, step=None, rate=None):
  global_step = self.global_step
  args = self.args
  lr = self.lr
  wd = self.wd
  sess = self.sess
  weight_decay = args.weight_decay
  if not args.learning_rate_cos:
    if step is None:
      step = eval_lightweight(global_step, session=sess)
    if rate is None:
      rate = args.learning_rate
    if callable(rate):
      rate = rate(step)
    load_lightweight(lr, rate, session=sess)
  load_lightweight(wd, weight_decay, session=sess)
  v_rate = eval_lightweight(lr, session=sess)
  v_weight_decay = eval_lightweight(wd, session=sess)
  return v_rate, v_weight_decay

tflex.trainer_update_lr = trainer_update_lr

def trainer_ensure(self):
  if self.init is not None:
    args = self.args
    self.say('Initializing...')
    self.sess.run(self.init, options=config_pb2.RunOptions(timeout_in_ms=tflex.initialize_timeout))
    if not args.fresh_model:
      tflex.load_trainer(self)
    self.say('Broadcasting variables...')
    tflex.trainer_reset_variables(self, self.all_vars, timeout_in_ms=5*60000)
    self.say('Warming up...')
    if not 'TFLEX_SKIP_WARMUP' in os.environ and not tflex.trainer_warmup(self):
      self.say('Warmup failed!')
      self.dead = True
      self.init = None
      return False
    else:
      self.say('Initialized.')
      self.init = None
  if not self.thread.is_alive():
    self.thread.start()
  return True

tflex.trainer_ensure = trainer_ensure

tflex.retry_count = 4

def trainer_warmup(self, retry_count=None, verbose=False):
  slices = tflex.trainer_slices(self)
  if retry_count is None:
    retry_count = tflex.retry_count
  # do a training step.
  i = 0
  while i < 5:
    i += 1
    self.say('Warmup: training step %d...' % i)
    for retry in range(retry_count):
      success = False
      try:
        tflex.trainer_fit(self, ignore=True)
        success = True
        break
      except DeadlineExceededError:
        pass
      if not success:
        return False
  if tflex.averaging:
    self.say('Warmup: averaging...')
    for index in tqdm.tqdm(range(slices)) if verbose else range(slices):
      for variables in [self.variables(index=index)]:
        success = False
        for retry in range(retry_count):
          try:
            # read the slice.
            self.say('Warmup: reading slice %d...' % index)
            values = self.sess.run(tflex.cast_variables(variables, graph=self.sess.graph), options=config_pb2.RunOptions(timeout_in_ms=tflex.read_deadline))
            # write the slice.
            self.say('Warmup: writing slice %d...' % index)
            tflex.trainer_assign_values(self, variables, values)
            success = True
            break
          except DeadlineExceededError:
            pass
        if not success:
          return False
  self.say('Warmup successful!')
  return True

tflex.trainer_warmup = trainer_warmup

def trainer_prepare(self):
  load_lightweight(self.global_step, self.counter, session=self.sess)
  v_rate, v_weight_decay = self.update_lr()
  return v_rate, v_weight_decay

tflex.trainer_prepare = trainer_prepare

def trainer_generate(self):
  self.say('Generating batch...')
  batch = self.sample_batch()
  print(repr(self.enc.decode(batch[0]))[0:150] + '...')
  return batch

tflex.trainer_generate = trainer_generate

def trainer_feed(self, batch):
  self.say('Loading context...')
  #load_lightweight(self.context, batch, session=self.sess, timeout_in_ms=tflex.context_load_timeout)
  feed = self.output['feed']
  return feed(batch, session=self.sess)

tflex.trainer_feed = trainer_feed

tflex.train_timeout = 1500000
tflex.gather_timeout = 240000
tflex.broadcast_timeout = 240000
tflex.loss_timeout = 3600000

def trainer_opt_apply(self, batch=None):
  if batch is None:
    batch = tflex.trainer_generate(self)
  #self.say('Running opt_feed...')
  tflex.trainer_feed(self, batch)
  self.say('Running opt_apply...')
  the = self.output['the']
  shards = self.output['shards']
  opt_losses = the.opt_losses
  opt_apply = the.opt_apply
  opt_gather = the.opt_gather
  opt_broadcast = the.opt_broadcast
  opt_train = the.opt_train
  #(_, v_loss, v_summary) = self.sess.run((self.opt_apply, self.loss, self.summaries), options=config_pb2.RunOptions(timeout_in_ms=self.timeout))
  #v_perp = math.exp(v_loss)
  losses = self.sess.run(opt_train, options=config_pb2.RunOptions(timeout_in_ms=tflex.train_timeout))
  v_losses = losses
  def thunk(_):
    nonlocal v_losses
    #self.say('Running opt_gather...')
    #self.sess.run(opt_gather, options=config_pb2.RunOptions(timeout_in_ms=tflex.gather_timeout))
    #self.say('Running opt_broadcast...')
    #self.sess.run(opt_broadcast, options=config_pb2.RunOptions(timeout_in_ms=tflex.broadcast_timeout))
    #self.say('Running opt_losses...')
    #v_losses = self.sess.run(opt_losses, options=config_pb2.RunOptions(timeout_in_ms=tflex.loss_timeout))
    #self.say('Loss deltas: %s' % (repr([x-y for x, y in zip(v_losses, losses)])))
    #batch_size = len(batch)
    #num_cores = len(shards)
    #assert(len(batch) % num_cores == 0)
    #j = batch_size // num_cores
    #parts = tflex.tuples(j, batch)
    #tflex.trainer_feed(self, parts[0]*num_cores)
    #losses = self.sess.run(opt_losses, options=config_pb2.RunOptions(timeout_in_ms=tflex.loss_timeout))
    #self.say('Losses (validation): %s before: %s' % (repr(losses), v_losses))
  #tflex.parallelize([0], thunk)
  thunk(0)
  tflex.trainer_flush(self)
  return v_losses

tflex.trainer_opt_apply = trainer_opt_apply

def trainer_flush(self):
  n = len(self.pending_writes)
  if n > 0:
    self.say('Flushing %d writes...' % n)
    start = time.time()
    i = 0
    while True:
      i += 1
      if not tflex.trainer_flush_once(self):
        break
    elapsed = time.time() - start
    self.say('Flushed %d writes in %.2fs' % (i, elapsed))

tflex.trainer_flush = trainer_flush

def trainer_flush_once(self):
  with self.lock:
    if len(self.pending_writes) <= 0:
      return False
    variables, values = self.pending_writes[0]
    self.pending_writes = self.pending_writes[1:]
  tflex.trainer_assign_values(self, variables, values)
  return True

tflex.trainer_flush_once = trainer_flush_once

def trainer_summary_log(self, v_loss):
  the = self.output['the']
  opt_loss = the.opt_loss
  v_summary = self.sess.run(self.summaries, feed_dict={opt_loss: v_loss}, options=config_pb2.RunOptions(timeout_in_ms=tflex.summary_log_timeout))
  return v_summary

tflex.trainer_summary_log = trainer_summary_log

def trainer_fit(self, ignore=False):
  v_rate, v_weight_decay = tflex.trainer_prepare(self)
  v_losses = tflex.trainer_opt_apply(self)
  v_loss = sum(v_losses) / len(v_losses)
  v_perp = math.exp(v_loss)
  if tflex.trainer_fresh(self):
    ignore = True
  if not ignore:
    v_summary = tflex.trainer_summary_log(self, v_loss)
    self.counter = self.current_step.incr(self.args.batch_size)
    self.summary_log.add_summary(v_summary, self.counter)
    self.summary_log.flush()
    self.avg_loss = [self.avg_loss[0] * 0.99 + v_loss,
                     self.avg_loss[1] * 0.99 + 1.0]
    self.avg_perp = [self.avg_perp[0] * 0.99 + v_perp,
                     self.avg_perp[1] * 0.99 + 1.0]
  now = time.time()
  print('{stamp} {target:16s}::{core} [{counter} | {time:2.4f} | {delta:2.2f}s | {ops:2.6f}tokens/s] loss={loss:2.4f}({avgloss:2.4f}) perp={perp:2.4f}({avgperp:2.4f}) lr={rate:0.12f} {flags}'#\n\tlosses={losses}'
      .format(
          stamp=timestamp(),
          core=self.core,
          target=self.sess.target[-16:],
          counter=self.counter,
          time=now - self.start_time,
          delta=now - self.prev_time,
          ops=self.args.sample_ctx * self.args.batch_size / (now - self.prev_time),
          rate=v_rate,
          loss=v_loss,
          perp=v_perp,
          avgloss=self.avg_loss[0] / (self.avg_loss[1] or 1.0),
          avgperp=self.avg_perp[0] / (self.avg_perp[1] or 1.0),
          flags='[fresh]' if ignore else '',
          losses=v_losses,
          ))
  self.prev_time = now
  #load_lightweight(self.global_step, self.counter, session=self.sess)
  return v_loss

tflex.trainer_fit = trainer_fit

def trainer_toplevel(self):
  while not self.stopped and tflex.trainer_alive(self):
    time.sleep(0.1)
    if self.paused:
      continue
    if not tflex.trainer_ensure(self):
      self.stopped = True
      break
    result = tflex.trainer_fit(self)
    if result is None or result is False:
      self.stopped = True
      break

tflex.trainer_toplevel = trainer_toplevel

def trainer_starting(trainer):
  if trainer.init:
    return True
  return False

tflex.trainer_starting = trainer_starting

def trainer_alive(trainer):
  if tflex.trainer_starting(trainer):
    return False
  if hasattr(trainer, "dead"):
    if trainer.dead:
      return False
  if not trainer.thread.is_alive():
    return False
  if trainer.sess.should_stop():
    return False
  return True

tflex.trainer_alive = trainer_alive

def trainer_fresh(trainer):
  return trainer_starting(trainer) or trainer.fresh

tflex.trainer_fresh = trainer_fresh

def reset_trainer_stats(trainer):
  x = trainer
  x.avg_loss[0] = x.avg_loss[1] = x.avg_perp[0] = x.avg_perp[1] = 0.0
  x.start_time = time.time()
  x.prev_time = x.start_time

tflex.reset_trainer_stats = reset_trainer_stats

def resume_trainer(trainer):
  if not tflex.trainer_alive(trainer):
    return False
  trainer.paused = False
  return True

tflex.resume_trainer = resume_trainer

def load_trainer(trainer, ckpt=None, reset_stats=True):
  args = trainer.args
  counter = trainer.counter
  saver = trainer.saver
  sess = trainer.sess
  trainer.say('Restoring...')
  if ckpt is None:
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
  with tflex.trainers_load_sema:
    saver.restore(sess, ckpt)
  t1 = time.time()
  print('Loaded in %f seconds' % (t1 - t0))
  if reset_stats:
    tflex.reset_trainer_stats(trainer)

tflex.load_trainer = load_trainer

def load_trainers(trainers=None, timeout=None):
  if trainers is None:
    trainers = list(tflex.get_trainers())
  trainers = [x for x in trainers if tflex.trainer_alive(x)]
  if timeout is None:
    timeout = len(trainers) * 30.0
  print('Loading %d trainers, max timeout %f' % (len(trainers), timeout))
  start_time = time.time()
  for thread in tqdm.tqdm(parallelize(trainers, tflex.load_trainer)):
    elapsed = (time.time() - start_time)
    waiting = timeout - elapsed
    if waiting > 0:
      thread.join(timeout=waiting)

tflex.load_trainers = load_trainers

def avgperp(trainer):
  return trainer.avg_perp[0] / (trainer.avg_perp[1] or 1.0)

tflex.avgperp = avgperp

def avgloss(trainer):
  return trainer.avg_loss[0] / (trainer.avg_loss[1] or 1.0)

tflex.avgloss = avgloss

def sorted_trainers(trainers=None):
  if trainers is None:
    trainers = [x for x in tflex.get_trainers()]
  return list(sorted(trainers, key=tflex.avgloss))

tflex.sorted_trainers = sorted_trainers

def print_trainer(x):
  ticks = 'ticks=%2.3f' % x.avg_loss[1]
  avgl = 'loss=%2.3f' % tflex.avgloss(x)
  avgp = 'perp=%2.3f' % tflex.avgperp(x)
  elapsed = 'elapsed=%ds' % int(x.prev_time - x.start_time)
  start = 'start=%d' % int(x.start_time)
  paused = 'paused=%s' % repr(x.paused)
  fresh = 'fresh=%s' % repr(tflex.trainer_fresh(x))
  alive = 'alive=%s' % repr(tflex.trainer_alive(x))
  print(x.sess.target, start, paused, fresh, alive, elapsed, avgl, avgp, ticks);
  return x

tflex.print_trainer = print_trainer

@tflex.register_command
def print_trainers(trainers=None):
  if trainers is None:
    trainers = list(tflex.get_trainers())
  trainers = [x for x in trainers if tflex.trainer_alive(x)]
  for x in tflex.sorted_trainers(trainers)[::-1]:
    tflex.print_trainer(x)
  print(len([x for x in trainers if not tflex.trainer_fresh(x) and tflex.trainer_alive(x)]), "trainers")

tflex.print_trainers = print_trainers

@tflex.register_command
def learning_rate_increase_5_percent():
  tflex.args.learning_rate *= 1.5
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_5_percent = learning_rate_increase_5_percent

@tflex.register_command
def learning_rate_decrease_5_percent():
  tflex.args.learning_rate /= 1.5
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_5_percent = learning_rate_decrease_5_percent

@tflex.register_command
def learning_rate_increase_10_percent():
  tflex.args.learning_rate *= 1.10
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_10_percent = learning_rate_increase_10_percent

@tflex.register_command
def learning_rate_decrease_10_percent():
  tflex.args.learning_rate /= 1.10
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_10_percent = learning_rate_decrease_10_percent

@tflex.register_command
def learning_rate_increase_20_percent():
  tflex.args.learning_rate *= 1.20
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_20_percent = learning_rate_increase_20_percent

@tflex.register_command
def learning_rate_decrease_20_percent():
  tflex.args.learning_rate /= 1.20
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_20_percent = learning_rate_decrease_20_percent

@tflex.register_command
def learning_rate_increase_2x():
  tflex.args.learning_rate *= 2
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_2x = learning_rate_increase_2x

@tflex.register_command
def learning_rate_decrease_2x():
  tflex.args.learning_rate /= 2
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_2x = learning_rate_decrease_2x

@tflex.register_command
def learning_rate_increase_5x():
  tflex.args.learning_rate *= 5
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_5x = learning_rate_increase_5x

@tflex.register_command
def learning_rate_decrease_5x():
  tflex.args.learning_rate /= 5
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_5x = learning_rate_decrease_5x

@tflex.register_command
def learning_rate_increase_10x():
  tflex.args.learning_rate *= 10
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_increase_10x = learning_rate_increase_10x

@tflex.register_command
def learning_rate_decrease_10x():
  tflex.args.learning_rate /= 10
  print('learning rate now %2.9f' % tflex.args.learning_rate)

tflex.learning_rate_decrease_10x = learning_rate_decrease_10x

def save_trainer(trainer):
  if not tflex.trainer_alive(trainer) or tflex.trainer_fresh(trainer):
    return False
  args = trainer.args
  counter = trainer.counter
  saver = trainer.saver
  sess = trainer.sess
  maketree(os.path.join(CHECKPOINT_DIR, trainer.args.run_name))
  print('Saving', os.path.join(CHECKPOINT_DIR, trainer.args.run_name, 'model-{}').format(counter))
  t0 = time.time()
  saver.save(sess, os.path.join(CHECKPOINT_DIR, args.run_name, 'model'), global_step=counter)
  t1 = time.time()
  print('Saved in %f seconds' % (t1 - t0))
  counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
  with open(counter_path, 'w') as fp:
      fp.write(str(counter) + '\n')
  return True

tflex.save_trainer = save_trainer

def rank_trainers(trainers=None):
  if trainers is None:
    trainers = [x for x in tflex.get_trainers()]
  return list(sorted(trainers, key=lambda x: x.avg_loss[1], reverse=True))

tflex.rank_trainers = rank_trainers

def save_trainers(trainers=None):
  for trainer in tflex.rank_trainers(trainers):
    print('-----')
    print('Saving:')
    print_trainer(trainer)
    print('-----')
    if save_trainer(trainer):
      print('-----')
      print_trainer(trainer)
      print('Saved')
      print('-----')
      return True
  return False

tflex.save_trainers = save_trainers

@tflex.register_command
def save_lowest_loss(trainers=None):
  for trainer in tflex.sorted_trainers(trainers):
    print('-----')
    print('Saving:')
    print_trainer(trainer)
    print('-----')
    if tflex.save_trainer(trainer):
      print('-----')
      print_trainer(trainer)
      print('Saved')
      print('-----')
      return True
  return False

tflex.save_trainers = save_lowest_loss

def parallelize(xs, thunk, *args):
  threads = []
  for x in xs:
    thread = threading.Thread(target=thunk, args=(x, *args))
    thread.start()
    threads.append(thread)
  return threads

#tflex.read_deadline = 20000
#tflex.write_deadline = 20000
tflex.read_deadline  = 120000
tflex.write_deadline = 240000
tflex.reset_deadline = 240000

def assign_values(variables, values, session=None, timeout_in_ms=tflex.write_deadline):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  session.run(ops, vals, options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms))

tflex.assign_values = assign_values

def trainer_reset_variables(self, variables, timeout_in_ms=tflex.reset_deadline):
  session = self.sess
  the = self.output['the']
  ops = [the.reset_var[v.name] for v in variables]
  session.run(ops, options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms))

tflex.trainer_reset_variables = trainer_reset_variables

def trainer_assign_values(self, variables, values, timeout_in_ms=tflex.write_deadline):
  tflex.assign_values(variables, values, session=self.sess, timeout_in_ms=timeout_in_ms)
  tflex.trainer_reset_variables(self, variables, timeout_in_ms=tflex.write_deadline)

tflex.trainer_assign_values = trainer_assign_values

def trainer_push_values(self, variables, values):
  with self.lock:
    self.pending_writes.append([variables, values])

tflex.trainer_push_values = trainer_push_values

class VariableAccumulator(object):
  pass

def variable_accumulator_new():
  self = VariableAccumulator()
  self.accum = {}
  self.accumcount = defaultdict(int)
  self.lock = threading.Lock()
  return self

tflex.variable_accumulator_new = variable_accumulator_new

def variable_accumulator_add(self, variable, value):
  if np.isnan(value).any():
    return False
  if np.isinf(value).any():
    return False
  if variable.name in self.accum:
    self.accum[variable.name] = self.accum[variable.name] + value
  else:
    self.accum[variable.name] = value
  self.accumcount[variable.name] += 1
  return True

tflex.variable_accumulator_add = variable_accumulator_add

def trainer_slice_read(trainer, accumulator, variables):
  values = trainer.sess.run(tflex.cast_variables(variables, graph=trainer.sess.graph), options=config_pb2.RunOptions(timeout_in_ms=tflex.read_deadline))
  with accumulator.lock:
    for variable, value in zip(variables, values):
      tflex.variable_accumulator_add(accumulator, variable, value)

tflex.trainer_slice_read = trainer_slice_read

tflex.trainer_slice_write_immediate = False if 'TFLEX_DELAY_WRITES' in os.environ else True

def trainer_slice_write(trainer, accumulator, variables):
  values = []
  for variable in variables:
    with accumulator.lock:
      assert(variable.name in accumulator.accum)
      value = accumulator.accum[variable.name]
      n = accumulator.accumcount[variable.name]
    assert(n > 0)
    values.append(value / n)
  if tflex.trainer_slice_write_immediate:
    tflex.trainer_assign_values(trainer, variables, values)
  else:
    tflex.trainer_push_values(trainer, variables, values)

tflex.trainer_slice_write = trainer_slice_write

tflex.update_trainers_read_timeout = 60
tflex.update_trainers_write_timeout = 60
tflex.update_trainers_write_threads = []

def update_trainers(trainers, i, sync_all=False):
  trainers = [x for x in trainers]
  if len(trainers) <= 0:
    return
  accumulator = tflex.variable_accumulator_new()
  threads = []
  for trainer in trainers:
    if tflex.trainer_fresh(trainer):
      continue
    def thunk(trainer, accumulator, index):
      for variables in ([trainer.variables(index=index)] if not sync_all else tqdm.tqdm(list(tflex.split_by_params(trainer.global_vars)))):
        tflex.trainer_slice_read(trainer, accumulator, variables)
    thread = threading.Thread(target=thunk, args=(trainer,accumulator,i,))
    thread.start()
    threads.append(thread)
  start_time = time.time()
  for thread in threads:
    elapsed = (time.time() - start_time)
    waiting = tflex.update_trainers_read_timeout - elapsed
    if waiting > 0:
      thread.join(timeout=waiting)
  start_time = time.time()
  for thread in tflex.update_trainers_write_threads:
    elapsed = (time.time() - start_time)
    waiting = tflex.update_trainers_write_timeout - elapsed
    if waiting > 0:
      thread.join(timeout=waiting)
  tflex.update_trainers_write_threads = []
  for trainer in trainers:
    def thunk(trainer, accumulator, index):
      for variables in ([trainer.variables(index=index)] if not sync_all else tqdm.tqdm(list(tflex.split_by_params(trainer.global_vars)))):
        tflex.trainer_slice_write(trainer, accumulator, variables)
    thread = threading.Thread(target=thunk, args=(trainer,accumulator,i,))
    thread.start()
    tflex.update_trainers_write_threads.append(thread)

tflex.update_trainers = update_trainers

def main():
    args = parser.parse_args()
    tflex.args = args
    tflex.enc = encoder.get_encoder(args.model_name)
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
    else:
        print('Unknown dtype', args.dtype)

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
      elif dataset.endswith('.tok16'):
        data_sampler = TokenSampler(dataset, enc, seed=seed, half=True)
      elif dataset.endswith('.tok32'):
        data_sampler = TokenSampler(dataset, enc, seed=seed, half=False)
      elif dataset.endswith('.tok'):
        assert not dataset.endswith('.tok')
        #data_sampler = TokenSampler(dataset, enc, seed=seed, half=False)
      else:
        data_sampler = TextSampler(dataset, enc, seed=seed, use_locking=True)
      return data_sampler

    tflex.make_sampler = make_sampler

    #print('Loading dataset...')
    #seed = None if args.seed < 0 else args.seed
    #tflex.data_sampler = make_sampler(dataset=args.dataset, enc=enc, seed=seed, combine=args.combine)

    print('Training...')
    counter = 1
    counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1

    local = threading.local()

    targets = [x.strip() for x in args.targets.split(',') if len(x.strip()) > 0]
    if len(targets) <= 0:
      targets.append('auto')
    tflex.targets = targets
    traincounter = TrainCounter(value=counter)
    tflex.trainers = []
    tflex.pending_trainers = []
    tflex.pinned_trainers = []
    tflex.trainers_sema = threading.BoundedSemaphore(value=60)
    tflex.trainers_init_sema = threading.BoundedSemaphore(value=6) # 150
    tflex.trainers_load_sema = threading.BoundedSemaphore(value=6) # 10
    tflex.trainers_lock = threading.RLock()
    while True:
      for target in tqdm.tqdm(tflex.targets, desc="Initializing first TPU..."):
        try:
          tflex.trainer = tflex.trainer_create(args=args, hparams=hparams, enc=tflex.enc, target=target, counter=traincounter)
          break
        except:
          import traceback
          traceback.print_exc()
          time.sleep(0.1)
    #tflex.trainer.ensure()
    #if not tflex.trainer.thread.is_alive():
    #  tflex.trainer.thread.start()
    #random.shuffle(tflex.targets)
    def add_trainer(target, delaying=10.0):
      #released = False
      try:
        if delaying > 0.0:
          time.sleep(random.random() * delaying)
        with tflex.trainers_lock:
          for existing in tflex.pending_trainers:
            if existing == target:
              return
          for existing in tflex.trainers:
            if existing.sess.target == target and tflex.trainer_alive(existing):
              return
          tflex.pending_trainers.append(target)
        try:
          with tflex.trainers_sema:
            #sampler = tflex.data_sampler
            #if not tflex.use_global_data_sampler:
            #  sampler = make_sampler(dataset=args.dataset, enc=enc, seed=seed, combine=args.combine)
            #trainer = tflex.trainer_create(args=args, hparams=hparams, sampler=sampler, enc=enc, target=target, counter=traincounter)
            trainer = tflex.trainer_fork(existing=tflex.trainer, target=target)
            #trainer.sampler = sampler
          tflex.pinned_trainers.append(trainer)
          #if tflex.release_trainer_sema:
          #  tflex.trainers_sema.release()
          #  released = True
          if tflex.ensure_on_init:
            with tflex.trainers_init_sema:
              trainer.ensure()
          with tflex.trainers_lock:
            for existing in tflex.trainers:
              if existing.sess.target == target:
                existing.stopped = True
                break
            if len(tflex.trainers) <= 0:
              print('Trainer %s is no longer fresh (first trainer)' % trainer.sess.target)
              trainer.fresh = False
            tflex.trainers.append(trainer)
        finally:
          tflex.pending_trainers.remove(target)
      finally:
        pass
        #if not released:
        #  tflex.trainers_sema.release()
    #start_time = time.time()
    #init_timeout = 10
    #for thread in tqdm.tqdm(parallelize(targets, add_trainer)):
    #  elapsed = (time.time() - start_time)
    #  waiting = init_timeout - elapsed
    #  if waiting > 0:
    #    thread.join(timeout=waiting)
    def add_trainers(targets=None):
      if targets is None:
        targets = tflex.targets
      for thread in tqdm.tqdm(parallelize(targets, add_trainer)):
        thread.join()
    tflex.adding_trainers = False
    def add_trainers_toplevel():
      while True:
        print('Re-adding all targets...')
        add_trainers()
        time.sleep(1.0)
        while not tflex.adding_trainers:
          time.sleep(1.0)
    tflex.add_swarm_thread = threading.Thread(target=add_trainers_toplevel)
    tflex.add_swarm_thread.start()
    #maxconnections = 2
    #tflex.trainers_sema = threading.BoundedSemaphore(value=maxconnections)
    #tflex.trainers[0].fresh = False

    def get_trainers():
      for trainer in tflex.trainers:
        if tflex.trainer_alive(trainer):
          if not tflex.trainer_starting(trainer):
            yield trainer

    tflex.get_trainers = get_trainers

    @tflex.register_command
    def save():
        maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
        save_trainers(tflex.get_trainers())

    #print("Warming up...")
    #def warmup(trainer):
    #  while trainer.current_step.value < 50:
    #    trainer.fit()
    #for thread in tqdm.tqdm(parallelize(tflex.get_trainers(), warmup)):
    #  thread.join()
    tflex.all_trainers = list(tflex.get_trainers())
    if args.fresh_model and len(tflex.all_trainers) > 1:
      print("Syncing...")
      tflex.update_trainers(tflex.all_trainers, 0, sync_all=True)
    print("Starting...")
    for trainer in tflex.get_trainers():
      print('Trainer %s is no longer fresh (startup trainers)' % trainer.sess.target)
      trainer.fresh = False
      #trainer.start()
    i = 0
    sync_thread = None
    first = True
    tflex.averaging_yield_time = 0.05 # was 1.0 # was 3.0
    tflex.averaging = not args.no_averaging
    tflex.cycle = None
    while True:
      tflex.check_commands()
      if tflex.should_quit():
        break
      tflex.all_trainers = list(tflex.get_trainers())
      threads = []
      #for trainer in tflex.all_trainers:
      #  def thunk(trainer, n):
      #    for _ in range(n):
      #      trainer.fit()
      #  count = 1 if first else 10
      #  thread = threading.Thread(target=thunk, args=(trainer,count))
      #  thread.start()
      #  threads.append(thread)
      #for thread in threads:
      #  thread.join()
      #print('Synchronizing...', i)
      #threads = []
      if len(tflex.all_trainers) <= 0:
        time.sleep(1.0)
      else:
        i += 1
        if not tflex.averaging:
          time.sleep(1.0)
        else:
          tflex.fresh_trainers = tflex.all_trainers[:]
          if tflex.cycle is None or tflex.shuffle_cycles:
            batches = len(tflex.all_trainers[0].fetch_vars)
            tflex.cycle = list(range(batches))
            random.shuffle(tflex.cycle)
          for index in tqdm.tqdm(tflex.cycle):
            tflex.check_commands()
            if tflex.should_quit():
              break
            tflex.all_trainers = list(tflex.get_trainers())
            tflex.fresh_trainers = [x for x in tflex.fresh_trainers if x in tflex.all_trainers]
            tflex.update_trainers(tflex.all_trainers, index)
            time.sleep(tflex.averaging_yield_time) # yield some CPU and network bandwidth
          for trainer in tflex.fresh_trainers:
            print('Trainer %s is no longer fresh' % trainer.sess.target)
            trainer.fresh = False
          first = False
          print('All done', i)


if __name__ == '__main__':
    main()

