import tensorflow as tf
from tensorflow import feature_column
def test_crossed_column():
    """ crossed column测试 """
    featrues = {
        'price': [['A'], ['B'], ['C']],
        'color': [['R'], ['G'], ['B']]
    }
    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    p_x_c = feature_column.crossed_column([price, color], 5)
    p_x_c_identy = feature_column.indicator_column(p_x_c)
    p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])

    price = feature_column.indicator_column(price)
    price = feature_column.input_layer(featrues, [price])

    color = feature_column.indicator_column(color)
    color = feature_column.input_layer(featrues, [color])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run([price]))
        print(sess.run([color]))
        print(sess.run([p_x_c_identy_dense_tensor]))

test_crossed_column()
###########################Hash function using############################################
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.tables_initializer())
# print(sess.run(tf.strings.to_hash_bucket_fast(["Hello", "TensorFlow", "2.x"], 5)))
# a = tf.constant([0,0,1])
# print(sess.run(tf.strings.to_hash_bucket_fast([a], 3)))

