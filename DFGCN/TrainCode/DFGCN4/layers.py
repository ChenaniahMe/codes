from inits import *
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim,placeholders,dropout=0.,sparse_inputs=False,
                 act=tf.nn.relu,bias=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.bias=bias
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        support = None
        pre_sup = dot(x, self.vars['weights_' + str(0)],
                      sparse=self.sparse_inputs)
        support = dot(self.support[0], pre_sup, sparse=True)
        return self.act(support)
