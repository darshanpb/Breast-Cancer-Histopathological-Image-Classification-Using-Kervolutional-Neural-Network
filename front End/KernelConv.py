from keras import backend as K
from keras.layers import Layer, Conv2D
from keras.constraints import Constraint, NonNeg
import tensorflow as tf

class Positive(Constraint):
  """Constrains the weights to be non-negative.
  """

  def __call__(self, w):
    return w * math_ops.cast(math_ops.greater(w, 0.), K.floatx())

class LinearKernel(Layer):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def call(self, x, w, b, data_format):
        outputs = K.dot(x, K.reshape(w, (-1, K.shape(w)[-1])))
        if b:
          outputs = K.bias_add(
                outputs,
                b,
                data_format=data_format)
        return outputs 

class PolynomialKernel(LinearKernel):
    def __init__(self, cp=2.0, dp=3, train_cp=True):
      assert(cp >= 0.0)
      self.cp = cp
      self.dp = dp
      self.train_cp = train_cp
      super(PolynomialKernel, self).__init__()

    def build(self, input_shape):
      self.cp = K.variable(self.cp, dtype='float64', constraint=NonNeg(), 
                           trainable=self.train_cp)
      super(PolynomialKernel, self).build(input_shape)

    def call(self, x, w, b, data_format):
        outputs = super(PolynomialKernel, self).call(x, w, b, data_format)
        return (self.cp + outputs)**self.dp

class RBFKernel(LinearKernel):
    def __init__(self, gamma=1.0, train_gamma=False):
      self.gamma = gamma
      self.train_gamma = train_gamma
      super(RBFKernel, self).__init__()

    def build(self, input_shape):
      self.gamma = K.variable(self.gamma, dtype='float64', 
                              trainable=self.train_gamma, constraint=NonNeg())
      super(RBFKernel, self).build(input_shape)

    def call(self, x, w, b, data_format):
        x = K.expand_dims(x)
        w = K.reshape(w, (-1, K.shape(w)[-1]))
        outputs = K.sum((x - w)**2, axis=-2)
        outputs = K.exp(outputs * -self.gamma)

        if b:
          outputs = K.bias_add(
                outputs,
                b,
                data_format=data_format)
        return outputs

class LaplaceRBFKernel(LinearKernel):
    def __init__(self, sigma=1.0):
      self.sigma = sigma
      super(LaplaceRBFKernel, self).__init__()

    def build(self, input_shape):
      self.sigma = K.variable(self.sigma, dtype='float64', 
                              trainable=self.train_sigma)
      super(RBFKernel, self).build(input_shape)

    def call(self, x, w, b, data_format):
        x = K.expand_dims(x)
        w = K.reshape(w, (-1, K.shape(w)[-1]))
        outputs = K.exp(-K.sqrt(K.sum((x - w)**2, axis=-2)) / self.sigma)

        if b:
          outputs = K.bias_add(
                outputs,
                b,
                data_format=data_format)
        return outputs

class HyperbolicKernel(LinearKernel):
    def __init__(self, k=1.0, c=-1.0, train_k=True, train_c=True):
      self.k = k
      self.c = c
      self.train_k = train_k
      self.train_c = train_c
      super(HyperbolicKernel, self).__init__()
      
    def build(self, input_shape):
      # k must be positive and c only negative
      self.k = K.variable(self.k, dtype='float64', 
                              trainable=self.train_k, constraint=Positive())
      self.c = K.variable(self.c, dtype='float64', 
                              trainable=self.train_c, constraint=Positive())
      super(HyperbolicKernel, self).build(input_shape)

    def call(self, x, w, b, data_format):
        outputs = super(HyperbolicKernel, self).call(x, w, b, data_format)
        return K.tanh(self.k * outputs - self.c)

class KernelConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format=None,
                 dilation_rate=1,
                 activation='relu', #None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_fn='linear',
                 **kwargs):
        if kernel_fn == 'linear':
            self.kernel_fn = PolynomialKernel()
        elif kernel_fn:
            self.kernel_fn = kernel_fn
        else:
            self.kernel_fn = None

        super(KernelConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, x):
        if not self.kernel_fn:
            return super(KernelConv2D, self).call(x)

        patches = tf.compat.v1.extract_image_patches(images=x,
                                           ksizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
                                           strides=[1, self.strides[0], self.strides[1], 1],
                                           rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
                                           padding=self.padding.upper())

        outputs = self.kernel_fn.call(x=patches, 
                                      w=self.kernel, 
                                      b=self.bias if self.use_bias else None,
                                      data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs