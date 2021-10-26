import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]

dfTrain = pd.read_csv(TRAIN_FILE)
dfTest = pd.read_csv(TEST_FILE)
# print(dfTrain.head(10))

df = pd.concat([dfTrain,dfTest])
feature_dict = {}
total_feature = 0
for col in df.columns:
    if col in IGNORE_COLS:
        continue
    elif col in NUMERIC_COLS:
        feature_dict[col] = total_feature
        total_feature += 1
    else:
        unique_val = df[col].unique()
        feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
        total_feature += len(unique_val)
print(total_feature)
print(feature_dict)


print(dfTrain.columns)
train_y = dfTrain[['target']].values.tolist()
dfTrain.drop(['target','id'],axis=1,inplace=True)
train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()

for col in train_feature_index.columns:
    if col in IGNORE_COLS:
        train_feature_index.drop(col,axis=1,inplace=True)
        train_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        train_feature_index[col] = feature_dict[col]
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1

print(dfTest.columns)

"""
对测试集进行转化
"""
test_ids = dfTest['id'].values.tolist()
dfTest.drop(['id'],axis=1,inplace=True)

test_feature_index = dfTest.copy()
test_feature_value = dfTest.copy()

for col in test_feature_index.columns:
    if col in IGNORE_COLS:
        test_feature_index.drop(col,axis=1,inplace=True)
        test_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        test_feature_index[col] = feature_dict[col]
    else:
        test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
        test_feature_value[col] = 1

import tensorflow as tf
import numpy as np
"""模型参数"""
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "eval_metric":'gini_norm',
    "random_seed":3
}
dfm_params['feature_size'] = total_feature
dfm_params['field_size'] = len(train_feature_index.columns)

"""开始建立模型"""
feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

"""建立weights"""
weights = dict()

# embeddings
weights['feature_embeddings'] = tf.Variable(
    tf.random_normal([dfm_params['feature_size'], dfm_params['embedding_size']], 0.0, 0.01),
    name='feature_embeddings')
weights['feature_bias'] = tf.Variable(tf.random_normal([dfm_params['feature_size'], 1], 0.0, 1.0), name='feature_bias')

# deep layers
num_layer = len(dfm_params['deep_layers'])
input_size = dfm_params['field_size'] * dfm_params['embedding_size']
glorot = np.sqrt(2.0 / (input_size + dfm_params['deep_layers'][0]))

weights['layer_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(input_size, dfm_params['deep_layers'][0])), dtype=np.float32
)
weights['bias_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][0])), dtype=np.float32
)

for i in range(1, num_layer):
    glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][i - 1] + dfm_params['deep_layers'][i]))
    weights["layer_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(dfm_params['deep_layers'][i - 1], dfm_params['deep_layers'][i])),
        dtype=np.float32)  # layers[i-1] * layers[i]
    weights["bias_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][i])),
        dtype=np.float32)  # 1 * layer[i]

# final concat projection layer

if dfm_params['use_fm'] and dfm_params['use_deep']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
elif dfm_params['use_fm']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size']
elif dfm_params['use_deep']:
    input_size = dfm_params['deep_layers'][-1]

glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                           dtype=np.float32)
weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

"""embedding"""
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)

reshaped_feat_value = tf.reshape(feat_value, shape=[-1, dfm_params['field_size'], 1])

embeddings = tf.multiply(embeddings, reshaped_feat_value)

"""fm part"""
fm_first_order = tf.nn.embedding_lookup(weights['feature_bias'], feat_index)
fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order, reshaped_feat_value), 2)

summed_features_emb = tf.reduce_sum(embeddings, 1)
summed_features_emb_square = tf.square(summed_features_emb)

squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

fm_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

"""deep part"""
y_deep = tf.reshape(embeddings, shape=[-1, dfm_params['field_size'] * dfm_params['embedding_size']])

for i in range(0, len(dfm_params['deep_layers'])):
    y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])
    y_deep = tf.nn.relu(y_deep)

"""final layer"""
if dfm_params['use_fm'] and dfm_params['use_deep']:
    concat_input = tf.concat([fm_first_order, fm_second_order, y_deep], axis=1)
elif dfm_params['use_fm']:
    concat_input = tf.concat([fm_first_order, fm_second_order], axis=1)
elif dfm_params['use_deep']:
    concat_input = y_deep

out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias']))

"""loss and optimizer"""
loss = tf.losses.log_loss(tf.reshape(label, (-1, 1)), out)
optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'], beta1=0.9, beta2=0.999,
                                   epsilon=1e-8).minimize(loss)

"""train"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        epoch_loss, _ = sess.run([loss, optimizer], feed_dict={feat_index: train_feature_index,
                                                               feat_value: train_feature_value,
                                                               label: train_y})
        print("epoch %s,loss is %s" % (str(i), str(epoch_loss)))