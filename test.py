import tensorflow as tf
import os
print(tf.distribute.cluster_resolver.TPUClusterResolver(os.environ['TPU_NAME']).get_master())

