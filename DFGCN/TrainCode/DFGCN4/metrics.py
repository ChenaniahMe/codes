import tensorflow as tf
def masked_softmax_cross_entropy(preds, labels, mask):
    #进行交叉熵的计算，logits是预测得到的结果，我们得到了交叉熵
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    #mask是一个列表
    mask = tf.cast(mask, dtype=tf.float32)
    #列表
    mask /= tf.reduce_mean(mask)
    #从以上可以loss是一个向量
    loss *= mask
    #最后的值为一个结果值
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """某一维上数据值最大的索引值，1表示以行为单位"""
    #相等为True,不相等为False
    result_mask=mask
    result_preds = preds
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    #这个进行变换后为0,1
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all),result_preds,result_mask