#https://blog.csdn.net/lin453701006/article/details/79391088
from __future__ import print_function
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

GENDER_COLUMNS = ["gender"]
LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

df_train = pd.read_csv(
    tf.gfile.Open("adult.data"),
    names=COLUMNS,
    skipinitialspace=True,
    engine="python"
)

df_train = df_train.dropna(how='any', axis=0)

df_train[LABEL_COLUMN] = (
    df_train["income_bracket"].apply(lambda x: ">50" in x).astype(int)
)

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in GENDER_COLUMNS}
    return continuous_cols
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
