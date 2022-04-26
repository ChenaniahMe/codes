from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def dot(x, y):
    res = tf.matmul(x, y)
    return res


class Layer(object):

    def __init__(self, **kwargs):
        self.vars = {}

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.act = act
        self.support = placeholders['support']
        self.vars['weights'] = glorot([input_dim, output_dim],
                                                name='weights')

    def _call(self, inputs):
        x = inputs
        pre_sup = dot(x, self.vars['weights'])
        support = dot(self.support, pre_sup)
        return self.act(support)
