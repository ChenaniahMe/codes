#!/usr/bin/env python
# coding: utf-8
# by Coggle数据科学

import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import utils
import warnings
from sklearn.utils import shuffle
import datapreprocess
warnings.filterwarnings('ignore')

# train_df = pd.read_csv('./temperature_data/train_set/train_testfusai_merge.csv')
train_df = pd.read_csv('../temperature_data/train_set/train_testfusai_merge.csv')
test_df = pd.read_csv('../temperature_data/test_set/test.csv')
sub1 = pd.read_csv('../fusai_code/raw_data/sub_fusai.csv')
# test_df = pd.read_csv('D:/bisai/temperature_data/test_set/test_addall.csv')
# test_df['outdoorHum'] = test_df['outdoorHum'].astype("float64")
# test_df['indoorHum'] = test_df['indoorHum'].astype("float64")
# test_df.pop('temperature')


train_df = train_df[train_df['temperature'].notnull()]
train_df = train_df.fillna(method='bfill')
# train_df = train_df.fillna(method='ffill')
test_df = test_df.fillna(method='bfill')
# test_df = test_df.fillna(method='ffill')


train_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                    'indoorHum', 'indoorAtmo', 'temperature']
test_df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 'outdoorTemp', 'outdoorHum', 'outdoorAtmo',
                   'indoorHum', 'indoorAtmo']
sub = pd.DataFrame(sub1['time'])
# train_df = utils.deal_outliers(train_df, ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo'])

data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 调用A
# data_df = data_df.iloc[:200,:]
data_df_A = utils.KNNThree(data_df[['outdoorTemp', 'outdoorHum', 'outdoorAtmo','indoorHum', 'indoorAtmo']],22)
data_df[['outdoorTemp', 'outdoorHum', 'outdoorAtmo','indoorHum', 'indoorAtmo']] = data_df_A[['outdoorTemp', 'outdoorHum', 'outdoorAtmo','indoorHum', 'indoorAtmo']]

train_count1 = train_df.shape[0]
train_df1 = data_df[:train_count1].copy().reset_index(drop=True)
test_df1 = data_df[train_count1:].copy().reset_index(drop=True)
test_df1.pop('temperature')
test1 = pd.DataFrame()
test = sub.merge(test_df1, how='left')
test1['time'] = test['time'].apply(datapreprocess.int2time) #apply(func [, args [, kwargs ]])函数用于当函数参数已经存在于
# 一个元组或字典中时，间接地调用函数。
# train['time'] = train['time'].apply(int2time)
test_new = pd.DataFrame()
test_new['year'] = test1['time'].map(lambda name:name.split('-')[0].strip())
test_new['month'] = test1['time'].map(lambda name:name.split('-')[1].strip())
test_new['month'] = test_new['month'].map(lambda name:name.split('0')[1].strip())
test_new['day'] = test1['time'].map(lambda name:name.split('-')[2].split(' ')[0].strip())
#test_new['day'] = test_new['day'].map(lambda name:name.split('0')[1].strip())
test_new['hour'] = test1['time'].map(lambda name:name.split(' ')[1].split(':')[0].strip())
test_new['min'] = test1['time'].map(lambda name:name.split(':')[1].strip())
test_new['sec'] = test1['time'].map(lambda name:name.split(':')[2].strip())

test['year'] = test_new['year']
test['month'] = test_new['month']
test['day'] = test_new['day']
test['hour'] = test_new['hour']
test['min'] = test_new['min']
test['sec'] = test_new['sec']

test['time'] = test['time'].astype("int64")
test['year'] = test['year'].astype("int64")
test['month'] = test['month'].astype("int64")
test['day'] = test['day'].astype("int64")
test['hour'] = test['hour'].astype("int64")
test['min'] = test['min'].astype("int64")
test['sec'] = test['sec'].astype("int64")

test_df2 = test.interpolate()
a = pd.DataFrame()
# # test_df2.fillna(train_df1.iloc[12728:12737, 7:11],inplace=True)
a = train_df1.iloc[12728:, 7:12].reset_index()
a.pop('index')
# a = a.reindex(index=['0','1','2','3','4','5','6','7','8','9'],columns=['outdoorTemp', 'outdoorHum', 'outdoorAtmo','indoorHum', 'indoorAtmo'])
test_df2.fillna(0, inplace=True)
test_df2['outdoorTemp'].iloc[0:10] = a['outdoorTemp'].map(lambda name:name)
test_df2['outdoorHum'].iloc[0:10] = a['outdoorHum'].map(lambda name:name)
test_df2['outdoorAtmo'].iloc[0:10] = a['outdoorAtmo'].map(lambda name:name)
test_df2['indoorHum'].iloc[0:10] = a['indoorHum'].map(lambda name:name)
test_df2['indoorAtmo'].iloc[0:10] = a['indoorAtmo'].map(lambda name:name)


data_df = pd.concat([train_df1, test_df2], axis=0, ignore_index=True)
# 基本聚合特征
group_feats = []
for f in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']):
    data_df['MDH_{}_medi'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('median')
    data_df['MDH_{}_mean'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('mean')
    data_df['MDH_{}_max'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('max')
    data_df['MDH_{}_min'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('min')
    data_df['MDH_{}_std'.format(f)] = data_df.groupby(['month', 'day', 'hour'])[f].transform('std')

    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))
    group_feats.append('MDH_{}_min'.format(f))
    group_feats.append('MDH_{}_max'.format(f))
    group_feats.append('MDH_{}_std'.format(f))


# 基本交叉特征
for f1 in tqdm(['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats):

    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo'] + group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data_df[colname] = data_df[f1].values / data_df[f2].values
            colname1 = '{}_{}_ratio1'.format(f1, f2)
            data_df[colname1] = data_df[f1].values - data_df[f2].values
            # colname2 = '{}_{}_ratio2'.format(f1, f2)
            # data_df[colname2] = (data_df[f1].values - data_df[f2].values) / data_df[f2].values

data_df = data_df.fillna(method='bfill')

# 历史信息提取
data_df['dt'] = data_df['day'].values + (data_df['month'].values - 3) * 31

for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo', 'temperature']:
    tmp_df = pd.DataFrame()
    for t in tqdm(range(15, 45)):
        tmp = data_df[data_df['dt'] < t].groupby(['hour'])[f].agg({'mean'}).reset_index()
        tmp.columns = ['hour', 'hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)

    data_df = data_df.merge(tmp_df, on=['dt', 'hour'], how='left')

data_df = data_df.fillna(method='bfill')
# data_df = data_df.fillna(method='ffill')

# 离散化
for f in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
    data_df[f + '_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_50_bin'] = pd.cut(data_df[f], 50, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_100_bin'] = pd.cut(data_df[f], 100, duplicates='drop').apply(lambda x: x.left).astype(int)
    data_df[f + '_200_bin'] = pd.cut(data_df[f], 200, duplicates='drop').apply(lambda x: x.left).astype(int)

for f1 in tqdm(
        ['outdoorTemp_20_bin', 'outdoorHum_20_bin', 'outdoorAtmo_20_bin', 'indoorHum_20_bin', 'indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(
        ['outdoorTemp_50_bin', 'outdoorHum_50_bin', 'outdoorAtmo_50_bin', 'indoorHum_50_bin', 'indoorAtmo_50_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_100_bin', 'outdoorHum_100_bin', 'outdoorAtmo_100_bin', 'indoorHum_100_bin',
                'indoorAtmo_100_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

for f1 in tqdm(['outdoorTemp_200_bin', 'outdoorHum_200_bin', 'outdoorAtmo_200_bin', 'indoorHum_200_bin',
                'indoorAtmo_200_bin']):
    for f2 in ['outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 'indoorAtmo']:
        data_df['{}_{}_medi'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1, f2)] = data_df.groupby([f1])[f2].transform('min')

# data_df['addfeature'] = (data_df['outdoorHum'].values / data_df['indoorHum'].values) * data_df['outdoorTemp'].values + \
#                         (data_df['outdoorAtmo'].values / data_df['indoorAtmo'].values) / 8.3


def single_model(clf, train_x, train_y, test_x, clf_name, class_num=1):
    #train = np.zeros((train_x.shape[0], class_num))
    #test = np.zeros((test_x.shape[0], class_num))

    nums = int(train_x.shape[0] * 0.80)

    if clf_name in ['sgd', 'ridge']:
        print('MinMaxScaler...')
        for col in features:
            ss = MinMaxScaler()
            ss.fit(np.vstack([train_x[[col]].values, test_x[[col]].values]))
            train_x[col] = ss.transform(train_x[[col]].values).flatten()
            test_x[col] = ss.transform(test_x[[col]].values).flatten()
            train_x[col] = train_x[col].fillna(0)
            test_x[col] = test_x[col].fillna(0)

    # 训练集验证集划分
    trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]

    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)
        data_matrix = clf.Dataset(train_x, label=train_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'mse',
            'min_child_weight': 5,
            'num_leaves': 2 ** 6,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'learning_rate': 0.003,
            'seed': 2020,
            'num_threads': 4,
        }
# 'num_threads': 4,
        model = clf.train(params, train_matrix, 60000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                          early_stopping_rounds=1000)
        model2 = clf.train(params, data_matrix, model.best_iteration)
        val_pred = model.predict(val_x, num_iteration=model2.best_iteration).reshape(-1, 1)
        test_pred = model.predict(test_x, num_iteration=model2.best_iteration).reshape(-1, 1)

    if clf_name == "xgb":
        train_matrix = clf.DMatrix(trn_x, label=trn_y, missing=np.nan)
        valid_matrix = clf.DMatrix(val_x, label=val_y, missing=np.nan)
        # test_matrix = clf.DMatrix(test_x, label=val_y, missing=np.nan)
        test_matrix = clf.DMatrix(test_x, missing=np.nan)
        params = {'booster': 'gbtree',
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 8,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'eta': 0.0015,
                  'seed': 2020,
                  'nthread': 4,
                  'silent': True,
                  }

        watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500,
                          early_stopping_rounds=1000)
        val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
        test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
# cat没变
    if clf_name == "cat":
        params = {'learning_rate': 0.0001, 'depth': 6, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=40000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "sgd":
        params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
        model = SGDRegressor(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "ridge":
        params = {
            'alpha': 1.3,
            'random_state': 2020,
        }
        model = Ridge(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    print("%s_mse_score:" % clf_name, mean_squared_error(val_y, val_pred))

    return val_pred, test_pred


def lgb_model(x_train, y_train, x_valid):
    lgb_train, lgb_test = single_model(lgb, x_train, y_train, x_valid, "lgb", 1)
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test


def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test


def sgd_model(x_train, y_train, x_valid):
    sgd_train, sgd_test = single_model(SGDRegressor, x_train, y_train, x_valid, "sgd", 1)
    return sgd_train, sgd_test


def ridge_model(x_train, y_train, x_valid):
    ridge_train, ridge_test = single_model(Ridge, x_train, y_train, x_valid, "ridge", 1)
    return ridge_train, ridge_test


drop_columns = ["time",  "sec", "temperature"]

train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)
# 打乱训练集
train_df = shuffle(train_df, random_state=2020)

features = train_df[:1].drop(drop_columns, axis=1).columns
x_train = train_df[features]
x_test = test_df[features]

# y_train = (train_df['temperature'].values - train_df['outdoorTemp'].values) / train_df['outdoorTemp'].values
y_train = train_df['temperature'].values / train_df['outdoorTemp'].values
# y_train = train_df['temperature'].values - train_df['outdoorTemp'].values
# y_train = np.log(train_df['temperature'].values) / np.log(train_df['outdoorTemp'].values)

#lr_train, lr_test = ridge_model(x_train, y_train, x_test)
#
# sgd_train, sgd_test = sgd_model(x_train, y_train, x_test)

#lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)

xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
#
# cat_train, cat_test = cat_model(x_train, y_train, x_test)

# train_pred = cat_train
# test_pred = cat_test

# sub["temperature"] = test_pred * test_df['outdoorTemp'].values
# sub.to_csv('sub_cattiaocan.csv', index=False)

# 五个模型平均
# train_pred = (lr_train + sgd_train + lgb_train[:, 0] + xgb_train[:, 0] + cat_train) / 5
# test_pred = (lr_test + sgd_test + lgb_test[:, 0] + xgb_test[:, 0] + cat_test) / 5


# 五个模型加权
# train_pred = 0.215335*lr_train + 0.06*sgd_train + 0.279625*lgb_train[:, 0] + 0.236986*xgb_train[:, 0] + 0.208054*cat_train
# test_pred = 0.215335*lr_test + 0.06*sgd_test + 0.279625*lgb_test[:, 0] + 0.236986*xgb_test[:, 0] + 0.208054*cat_test

#去掉sgd加权
#train_pred = 0.255571*lr_train + 0.258976*lgb_train[:, 0] + 0.262789*xgb_train[:, 0] + 0.222664*cat_train
#test_pred = 0.255571*lr_test + 0.258976*lgb_test[:, 0] + 0.262789*xgb_test[:, 0] + 0.222664*cat_test

# 减法预测
# sub["temperature"] = lgb_test[:, 0] + test_df['outdoorTemp'].values
# sub.to_csv('sub_082_KNNthree_lgbjianhaoyuce.csv', index=False)

# KNNThreelgb 0.10366
# sub["temperature"] = lgb_test[:, 0] * test_df['outdoorTemp'].values
# sub.to_csv('sub_082_KNNthree_lgb.csv', index=False)

# 分子分母都取对数然后用比值进行预测
# sub["temperature"] = np.exp(lgb_test[:, 0] * np.log(test_df['outdoorTemp'].values))
# sub.to_csv('sub_082_KNNthree_lgb_duishubizhi.csv', index=False)

# KNNThreelgb 0.10366，使用sgd,lr,cat,xgb的结果当作特征
# sub["temperature"] = sgd_train * train_df['outdoorTemp'].values
# sub.to_csv('sub_082_KNNthree_sgd_train.csv', index=False)

# KNNThreelgb (a-b)/b进行预测 交叉特征加了a-b
# sub["temperature"] = lgb_test[:, 0] * test_df['outdoorTemp'].values + test_df['outdoorTemp'].values
# sub["temperature"] = lgb_test[:, 0] * test_df['outdoorTemp'].values
# sub.to_csv('D:/bisai/temperature_data/fusai_080_Algb_bizhi_testaddliner_tiaocan.csv', index=False)

#train_pred = (lgb_train[:, 0] + xgb_train[:, 0] + cat_train) / 3
#test_pred = (lgb_test[:, 0] + xgb_test[:, 0] + cat_test) / 3
#sub["temperature"] = test_pred * test_df['outdoorTemp'].values
#sub.to_csv('./temperature_data/fusai_080_3model_bizhi_trainadd_testaddlinercan.csv', index=False)
sub["temperature"] =xgb_test[:, 0] * test_df['outdoorTemp'].values
sub.to_csv('./1.csv', index=False)
'''
train_pred = lr_train
test_pred = lr_test

sub["temperature"] = test_pred * test_df['outdoorTemp'].values
sub.to_csv('sub_lr_tiaocan.csv', index=False)

train_pred1 = sgd_train
test_pred1 = sgd_test

sub["temperature"] = test_pred1 * test_df['outdoorTemp'].values
sub.to_csv('sub_sgd_tiaocan.csv', index=False)

train_pred = lgb_train[:, 0]
test_pred = lgb_test[:, 0]

sub["temperature"] = test_pred * test_df['outdoorTemp'].values
sub.to_csv('sub_lgb.csv', index=False)

train_pred1 = xgb_train[:, 0]
test_pred1 = xgb_test[:, 0]

sub["temperature"] = test_pred1 * test_df['outdoorTemp'].values
sub.to_csv('sub_xgb.csv', index=False)

train_pred2 = cat_train
test_pred2 = cat_test

sub["temperature"] = test_pred2 * test_df['outdoorTemp'].values
sub.to_csv('sub_cat.csv', index=False)
'''
