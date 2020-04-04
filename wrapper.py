import tensorflow as tf
import sys

def uncache(exclude):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.
    
    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
      del sys.modules[mod]

#from tflex import TPUClusterResolver as FlexTPUClusterResolver
import re
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver as BaseTPUClusterResolver
from tensorflow.python.training import server_lib

def reroute(addr, host=None):
  if host is None or host is False:
    return addr
  if addr.startswith('grpc://'):
    return 'grpc://' + reroute(addr[len('grpc://'):], host=host)
  if not re.match('[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+[:]8470', addr):
    return addr
  if not addr.endswith(':8470'):
    return addr
  a, b, c, d = [int(x) for x in addr.split(':')[0].split('.')]
  if a == 10 and b in [48, 49]:
    assert (d == 2)
    port = b * 1000 + c
  elif a == 10 and b in range(2, 66) and c == 0:
    port = b * 1000 + d
  else:
    return addr
  return host + ':' + str(port)


class TPUClusterResolver(BaseTPUClusterResolver):
  def __init__(self, *args, host=None, **kws):
    super(TPUClusterResolver, self).__init__(*args, **kws)
    print('HERE')
    if host is None:
      if 'TPU_HOST' in os.environ:
        host = os.environ['TPU_HOST']
    self._host = host

  def master(self, *args, **kws):
    ip = super(TPUClusterResolver, self).master(*args, **kws)
    print('TKTK')
    return reroute(ip, host=self._host)

  def cluster_spec(self):
    spec = super(TPUClusterResolver, self).cluster_spec()
    r = dict()
    for k, v in spec.as_dict().items():
      r[k] = [reroute(ip, host=self._host) for ip in v]
    return server_lib.ClusterSpec(r)

import os
tf.python.distribute.cluster_resolver.TPUClusterResolver = TPUClusterResolver
#uncache('tensorflow.python.distribute.cluster_resolver.TPUClusterResolver')
tf.distribute.cluster_resolver.TPUClusterResolver = TPUClusterResolver
#uncache('tensorflow.distribute.cluster_resolver.TPUClusterResolver')
tf.contrib.cluster_resolver.TPUClusterResolver = TPUClusterResolver
#uncache('tensorflow.contrib.cluster_resolver.TPUClusterResolver')
#print(TPUClusterResolver(os.environ['TPU_NAME']).get_master())
if __name__ == '__main__':
  sys.argv = sys.argv[1:]
  exec(open(sys.argv[0]).read())
