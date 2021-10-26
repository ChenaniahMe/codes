#https://blog.csdn.net/pearl8899/article/details/107958334
#https://zhuanlan.zhihu.com/p/73701872
import tensorflow as tf
#****************************one**************************************
'''
sess = tf.Session()
# 特征数据
features = {
    'department': [10,20,30,40,50,17,64],
}

# 特征列
# department = tf.feature_column.categorical_column_with_hash_bucket('department', 5, dtype=tf.string)
# department = tf.contrib.layers.sparse_column_with_keys('department',  keys=["female","male"], dtype=tf.string)
department = tf.contrib.layers.real_valued_column('department')
department = tf.contrib.layers.bucketized_column(department, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# department = tf.feature_column.indicator_column(department)
# 组合特征列
columns = [department]
# 输入层（数据，特征列） 
inputs = tf.feature_column.input_layer(features, columns)
# 初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)
v = sess.run(inputs)
print(v)
'''

#****************************two**************************************
'''
import tensorflow as tf
sess=tf.Session()

#特征数据
features = {
    # 'sex': [1, 2, 1, 1, 2],
    'education': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
    'department': ['sport', 'drawing', 'drawing', 'travelling', 'travelling'],
}

#特征列
department = tf.contrib.layers.sparse_column_with_hash_bucket("department", hash_bucket_size = 5)
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size = 5)
# department = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sport','drawing','gardening','travelling'], dtype=tf.string)
# sex = tf.feature_column.categorical_column_with_identity('sex', num_buckets=2, default_value=0)

edu_department = tf.feature_column.crossed_column([department,education], 5)
edu_department = tf.feature_column.indicator_column(edu_department)
department = tf.feature_column.indicator_column(department)
education = tf.feature_column.indicator_column(education)
#组合特征列
columns_one = [
    edu_department
]
columns_two = [
    department
]
columns_three = [
    education
]
#输入层（数据，特征列）
inputs_one = tf.feature_column.input_layer(features, columns_one)
inputs_two = tf.feature_column.input_layer(features, columns_two)
inputs_three = tf.feature_column.input_layer(features, columns_three)
#初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)
print(sess.run(inputs_one))
print(sess.run(inputs_two))
print(sess.run(inputs_three))
'''
import tensorflow as tf
sess=tf.Session()

#特征数据
features = {
    'sex': [1, 2, 1, 1, 2],
    'department': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
}

#特征列
# department = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sport','drawing','gardening','travelling'], dtype=tf.string)
department = tf.contrib.layers.sparse_column_with_hash_bucket("department", hash_bucket_size = 5)
sex = tf.feature_column.categorical_column_with_identity('sex', num_buckets=5, default_value=0)
sex_department = tf.feature_column.crossed_column([department,department], 5)
department = tf.feature_column.indicator_column(department)
sex = tf.feature_column.indicator_column(sex)
sex_department = tf.feature_column.indicator_column(sex_department)
#组合特征列
columns_one = [
    department
]

columns_two = [
    department
]
columns = [
    sex_department
]

#输入层（数据，特征列）
inputs_one = tf.feature_column.input_layer(features, columns_one)
inputs_two = tf.feature_column.input_layer(features, columns_two)
inputs = tf.feature_column.input_layer(features, columns)

#初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)

print(sess.run(inputs_one))
print(sess.run(inputs_two))
print(sess.run(inputs))

