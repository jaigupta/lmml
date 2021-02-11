"""Big GANS keras layer."""

import math
import tensorflow as tf

class ConditionalBatchNorm(tf.keras.layers.Layer):
  """Conditional batch norm."""

  def __init__(self, decay=0.9, epsilon=1e-5):
    super().__init__()
    self.decay = decay
    self.epsilon = epsilon

  def build(self, xz_shape):
    x_shape, _ = xz_shape
    self.c = x_shape[-1]
    self.fc_beta = tf.keras.layers.Dense(self.c)
    self.fc_gamma = tf.keras.layers.Dense(self.c)

    self.test_mean = tf.Variable(tf.zeros((self.c,)), trainable=False)
    self.test_var = tf.Variable(tf.zeros((self.c,)), trainable=False)

  def call(self, xz, training=False):
    x, z = xz
    beta = tf.reshape(self.fc_beta(z), (-1, 1, 1, self.c))
    gamma = tf.reshape(self.fc_gamma(z), (-1, 1, 1, self.c))
    if training:
      batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
      self.test_mean.assign(self.test_mean * self.decay + batch_mean *
                            (1 - self.decay))
      self.test_var.assign(self.test_var * self.decay + batch_var *
                           (1 - self.decay))
      return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma,
                                       self.epsilon)
    else:
      return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta,
                                       gamma, self.epsilon)


class Deconv(tf.keras.layers.Layer):
  """Deconvolution layer."""

  def __init__(self, channels, kernel_size, strides=(1, 1), padding='same'):
    super().__init__()
    self.conv2dt = tf.keras.layers.Conv2DTranspose(
        channels,
        kernel_size,
        strides,
        padding,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0,
            stddev=1 /
            math.sqrt(kernel_size[0] * kernel_size[1] * channels * 1.2)))

  def call(self, x):
    return self.conv2dt(x)


class ResBlockUp(tf.keras.layers.Layer):
  """Residual block to increase resolution."""

  def __init__(self, channels):
    super().__init__()
    self.batch_norm1 = ConditionalBatchNorm()
    self.deconv1 = Deconv(channels, kernel_size=(3, 3), strides=(2, 2))
    self.batch_norm2 = ConditionalBatchNorm()
    self.deconv2 = Deconv(channels, kernel_size=(3, 3), strides=(1, 1))
    self.shortcut_deconv = Deconv(channels, kernel_size=(3, 3), strides=(2, 2))

  def call(self, x, z, training=False):
    o = self.batch_norm1((x, z), training=training)
    o = tf.nn.relu(o)
    o = self.deconv1(o)

    o = self.batch_norm2((o, z), training=training)
    o = tf.nn.relu(o)
    o = self.deconv2(o)

    o_shortcut = self.shortcut_deconv(x)
    return o + o_shortcut


class SelfAttention(tf.keras.layers.Layer):
  """Self attention layer on convolution."""

  def build(self, x_shape):
    ch = x_shape[3]
    self.conv_f = tf.keras.layers.Conv2D(
        ch // 8, kernel_size=(3, 3), strides=(1, 1), padding='same')
    self.pool_f = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='same')

    self.conv_g = tf.keras.layers.Conv2D(
        ch // 8, kernel_size=(3, 3), strides=(1, 1), padding='same')

    self.conv_h = tf.keras.layers.Conv2D(
        ch // 2, kernel_size=(3, 3), strides=(1, 1), padding='same')
    self.pool_h = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='same')

    self.final_conv = tf.keras.layers.Conv2D(
        ch, kernel_size=(1, 1), strides=(1, 1), padding='same')

    self.gate = tf.Variable(1.0)

  def call(self, x):
    bs, sh, sw, c = x.shape
    f = self.conv_f(x)
    f = self.pool_f(f)

    g = self.conv_g(x)

    h = self.conv_h(x)
    h = self.pool_h(h)

    attention = tf.matmul(
        tf.reshape(g, (bs, sh * sw, c // 8)),
        tf.reshape(f, (bs, sh * sw // 4, c // 8)),
        transpose_b=True)
    attention = tf.nn.softmax(attention, axis=-1)
    weighted_output = attention @ tf.reshape(h, (bs, sh * sw // 4, c // 2))
    weighted_output = tf.reshape(weighted_output, (bs, sh, sw, c // 2))
    output = self.final_conv(weighted_output)
    return x + self.gate * output


class BigGan(tf.keras.layers.Layer):
  """Big GANS layer."""

  def __init__(self):
    super().__init__()
    self.fc_z = tf.keras.layers.Dense(
        4 * 4 * 128, kernel_initializer='glorot_normal')
    self.resup1 = ResBlockUp(256)
    self.resup2 = ResBlockUp(128)
    self.resup3 = ResBlockUp(64)
    self.resup4 = ResBlockUp(32)
    self.resup5 = ResBlockUp(16)
    self.sa = SelfAttention()
    self.resup6 = ResBlockUp(8)

    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
    self.conv = tf.keras.layers.Conv2D(
        3, kernel_size=(3, 3), strides=(1, 1), padding='same')

  def call(self, z, training=False):
    z = tf.split(z, 7, axis=-1)
    x = self.fc_z(z[0])
    x = tf.reshape(x, (-1, 4, 4, 128))
    x = self.resup1(x, z[1], training=training)
    x = self.resup2(x, z[2], training=training)
    x = self.resup3(x, z[3], training=training)
    x = self.resup4(x, z[4], training=training)
    x = self.resup5(x, z[5], training=training)
    x = self.sa(x)
    x = self.resup6(x, z[6], training=training)
    x = x[:, 16:-16, 16:-16, :]

    x = self.bn(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv(x)

    x = tf.math.tanh(x)

    return x
