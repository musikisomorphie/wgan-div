import numpy as np
import tensorflow as tf

import tflib as lib
from tflib.sn import spectral_normed_weight



def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", spectral_normed=True, update_collection=None, with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    # if scope_has_variables(scope):
    #   scope.reuse_variables()
    w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                          strides=[1, d_h, d_w, 1], padding=padding)
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv