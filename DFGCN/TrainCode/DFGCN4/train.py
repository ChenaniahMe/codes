from __future__ import division
from __future__ import print_function
import time
from utils import *
from models import GCN
import numpy as np
import config
import pandas as pd
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', r'data/', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('train_rate', 0.1, 'traing rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('num_classes', len(config.numClass), 'the number of classes labeles')
flags.DEFINE_float('weight_decay', 5e-4, 'the weight the loss of L2')
flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer xth.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('layers', 2, 'the number of layers is equal to layers, is needed as 2')

adj, features, train_label, y_test, mask_train, mask_train1,mask_test, y_testOld, mask_testOld = load_data(FLAGS.dataset_path)

features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
placeholders = {
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels': tf.placeholder(tf.float32, shape=(None, train_label.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'support': [tf.sparse_placeholder(tf.float32)],
    'support_one': [tf.sparse_placeholder(tf.float32)],
    'support_two': [tf.sparse_placeholder(tf.float32)],
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# print("调用模型开始,网络层数为",FLAGS.layers)
model = model_func(placeholders, input_dim=features[2][1],logging=True)
sess = tf.Session()

def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,model.result_pvalue, model.result_mask], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],(time.time() - t_test), outs_val[2],outs_val[3]

sess.run(tf.global_variables_initializer())

epochStart = time.time()
result_pvalue=None
result_mask=None
for epoch in range(FLAGS.epochs):
    #train
    t = time.time()
    feed_dict = construct_feed_dict(features, support, train_label, mask_train, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # test
    test_cost, test_acc, test_duration,result_pvalue, result_mask= evaluate(features, support, y_testOld, mask_testOld, placeholders)

    #print
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t), "test_loss=","{:.5f}".format(test_cost))
#ALL Test
result_pvalue = pd.DataFrame(result_pvalue).to_csv("result_pvalue.csv")
result_mask = pd.DataFrame(result_mask).to_csv("result_mask.csv")
print(result_pvalue)
print(result_mask)
#Each Class Test
epochEnd = time.time()
costTime = epochEnd- epochStart
#To save results about loss of both train and test

print("Optimization Finished!")
#Test models
result=[]
for i in range(0,len(config.numClass)):
    test_cost, test_acc, test_duration, result_pvalue, result_mask= evaluate(features, support,y_test[i], mask_test[i], placeholders)
    result.append(test_acc)
    print("Test set results in {} class:".format(i), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
t_result=[]
for i in range(0,len(result)):
    if(np.isnan(result[i])==False):
        t_result.append(result[i])
        print(result[i], "\t", end="")
    else:
        t_result.append(0)

result=t_result
OA=sum(np.array(config.numClass)*np.array(result))/sum(config.numClass)
AA=sum(np.array(result))/len(config.numClass)
Pe=sum(np.array(config.numClass)*np.array(config.numClass)*np.array(result))/sum(config.numClass)**2
Po=sum(np.array(config.numClass)*np.array(result))/sum(config.numClass)
Kappa=(Po-Pe)/(1-Pe)
print("OA:", OA, "AA:",AA, "Kappa:", Kappa, "Time:", costTime )
