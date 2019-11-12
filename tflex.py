import tensorflow as tf
import numpy as np
from glob import glob
import os
import re
from tensorflow.python import pywrap_tensorflow
import tqdm
import h5py
import shutil
import tempfile

def split_by_params(vs, n=200e6, f=None):
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def latest_checkpoint(checkpoint_dir, latest_filename=None):
  ctrs = np.array([[int(y) for y in re.findall(r'model-([0-9]+)(?:-[0-9]+)?[.](?:npy|hdf5)', x)] for x in glob(os.path.join(checkpoint_dir, 'model-*.*'))]).flatten()
  if len(ctrs) <= 0:
    ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=latest_filename)
    return ckpt
  ctr = ctrs.max()
  return os.path.join(checkpoint_dir, 'model-{}').format(ctr)

def truncate_value(variable, value):
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  print('Truncating {} from shape {} to shape {}'.format(variable.name, value.shape, shape))
  value = value.reshape([-1])
  value = value[0:params]
  value = value.reshape(shape)
  return value

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable.name.split(':')[0]
    value = reader.get_tensor(name)
    if reshape:
      value = truncate_value(variable, value)
    yield variable, value

def assign_values(variables, values, session=None):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  session.run(ops, vals)

def load_snapshot(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape)]
    assign_values(variables, values, session=session)

def get_variable(name, var_list=None):
  name, num = name.split(':') if ':' in name else (name, '0')
  num = int(num)
  name = os.path.join(tf.get_variable_scope().name, name)
  vs = var_list or tf.trainable_variables()
  for x in vs:
      if x.name.startswith(name + ':%d' % num):
          return x

def load_weights(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  vs = var_list or tf.trainable_variables()
  files = list(sorted(glob(ckpt + '-*.npy')))
  for out in tqdm.tqdm(files):
    for name, value in np.load(out, allow_pickle=True):
      variable = get_variable(name)
      if variable is None:
        print('Warning: variable %s not loaded' % name)
      else:
        if reshape:
          value = truncate_value(variable, value)
        variable.load(value, session)

def load_variables(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  vs = var_list or tf.trainable_variables()
  with h5py.File(ckpt) as f:
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = [f[x.name] for x in variables]
      assign_values(variables, values, session=session)

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def save_variables(ckpt, session=None, var_list=None):
    session = session or tf.get_default_session()
    vs = var_list or tf.trainable_variables()
    with tempfile.TemporaryDirectory() as base:
      fname = os.path.join(base, 'model.hdf5')
      with h5py.File(fname, "w") as f:
        for variables in tqdm.tqdm(list(split_by_params(vs))):
          values = session.run(variables)
          for value, variable in zip(values, variables):
            name = variable.name
            shape = variable.shape.as_list()
            dtype = variable.dtype
            dset = f.create_dataset(name, shape, dtype=np.float32)
            dset[:] = value
      print('Writing snapshot...')
      maketree(os.path.dirname(ckpt))
      shutil.copyfile(fname, ckpt+'.tmp')
      os.rename(ckpt+'.tmp', ckpt)

class Saver(object):
  def __init__(
    self,
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None):
    self.var_list = var_list
    self.reshape = reshape
    self.sharded = sharded
    self.max_to_keep = max_to_keep
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self.name = name
    self.restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self.builder = builder
    self.defer_build = defer_build
    self.allow_empty = allow_empty
    self.write_version = write_version
    self.pad_step_number = pad_step_number
    self.save_relative_paths = save_relative_paths
    self.filename = filename
    self.checkpoints = []

  def restore(self, sess, save_path):
    if save_path.endswith('.ckpt'):
      load_snapshot(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif save_path.endswith('.hdf5'):
      load_variables(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.hdf5'):
      load_variables(save_path + '.hdf5', session=sess, var_list=self.var_list, reshape=self.reshape)
    else:
      load_weights(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)

  def save(self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False,
        save_debug_info=False):
    if global_step is not None:
      name = '{}-{}.hdf5'.format(save_path, global_step)
    else:
      name = '{}.hdf5'.format(save_path)
    save_variables(name, session=sess, var_list=self.var_list)
    self.checkpoints.append(name)
    if self.max_to_keep > 0:
      while len(self.checkpoints) > self.max_to_keep:
        fname = self.checkpoints[0]
        if fname != name:
          with open(fname, "w") as f:
            pass
        self.checkpoints = self.checkpoints[1:]

    
