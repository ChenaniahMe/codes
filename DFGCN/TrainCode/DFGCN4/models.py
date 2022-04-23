from layers import *
from metrics import *
flags = tf.app.flags
FLAGS = flags.FLAGS
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name=kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.result_pvalue=None
        self.result_mask=None
        self.optimizer = None
        self.opt_op = None

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        self.activations.append(self.inputs)
        tempI=1
        tempOut=[]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            if tempI > 1 and tempI<len(self.layers):
                hidden = hidden + tempOut[-1]
            if tempI == 15:
                hidden = tempOut[6]+tempOut[7]+tempOut[8]+tempOut[9]+tempOut[10]+tempOut[11]+tempOut[12]+tempOut[13]
            tempI += 1
            tempOut.append(hidden)
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs=placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.train_rate)
        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy,self.result_pvalue,self.result_mask = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        temp_layers_num=FLAGS.layers-2
        for i in range(0,temp_layers_num):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

