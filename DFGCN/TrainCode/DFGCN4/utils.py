import scipy.sparse as sp
import os
import numpy as np
import config
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def sample_mask(idx, l):
    # 创建了一个1行但是L个元素的0向量
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_lables(labels_train):
    tempLabel = []
    for i in range(len(labels_train)):
        temp = []
        for j in range(1, FLAGS.num_classes + 1):
            if labels_train[i] == j:
                temp.append(1)
            else:
                temp.append(0)

        tempLabel.append(temp)
    return np.array(tempLabel)


def load_data(dataset_path):
    ''' load data and return the data of model
    Args:
        dataset_path: the path of dataset
    Return:
        adj: the adj processed
        features: the feature of data
        y_train:
        y_testArr:
        mask_train:
        mask_testArr:
        y_testOld:
        mask_testOld:
    '''
    """Load  dataset """
    # file_name="T3-Nonfiltered-SmallFarm"
    # file_name = "Data/python/"
    file_name = "SmallFlevoland/python/"
    # file_name = "ReduceSmallFlevoland/python/"
    file_features="/data/ztw/PolSARPubPaperOne/Experiment/DealData/"+file_name+"Feature.npy"
    file_label="/data/ztw/PolSARPubPaperOne/Experiment/DealData/"+file_name+"label.npy"
    file_adj="/data/ztw/PolSARPubPaperOne/Experiment/DealData/"+file_name+"adj.npz"
    # file_features="./data/Feature.npy"
    # file_features="./data/Feature.npy"
    # file_label="./data/label.npy"
    # file_adj="./data/adj300.npz"
    reduce_index = np.arange(0, 2000, 1)
    features = np.load(file_features)
    labels = np.load(file_label)
    # # reduce features matrix
    # features_one = features[reduce_index, :]

    train_rate = FLAGS.train_rate
    C = FLAGS.num_classes  # the number of each class
    # idx_ev_class是一个字典，每一个元素表示的是这个类所占的位置
    num_ev_class, idx_ev_class = get_class_number(C, labels)
    # 得到训练集对应的索引
    idx_train, idx_test = create_index(C, num_ev_class, idx_ev_class, train_rate)
    # idx_test = idx_test[labels[idx_test] == 6]
    idx_testArr = []
    each_class=config.each_class

    for i in range(1, FLAGS.num_classes + 1):
        # i=int(i)
        # if i!=0:
        idx_testArr.append(idx_test[labels[idx_test] == i])
    adj = sp.load_npz(file_adj)
    # 将adj变为了对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    #reduce graph matrix
    adj_one = adj.tocsr()
    adj_one = adj_one[reduce_index, :]
    adj_one = adj_one[:, reduce_index]
    adj_one = sp.coo_matrix(adj_one)

    labels = preprocess_lables(labels)
    mask_train = sample_mask(idx_train, labels.shape[0])  # idx_train is index of traning data
    mask_train1= np.array([1]*len(idx_train), dtype=np.bool)
    mask_testOld = sample_mask(idx_test, labels.shape[0])  # idx_test is index of test data
    # test arruarcy for each class
    y_testArr = []
    mask_testArr = []
    for i in range(1, FLAGS.num_classes + 1):
        mask_test = sample_mask(idx_testArr[i - 1], labels.shape[0])
        y_test = np.zeros(labels.shape)
        y_test[mask_test, :] = labels[mask_test, :]
        y_testArr.append(y_test)
        mask_testArr.append(mask_test)
    train_label = np.zeros(labels.shape)
    y_testOld = np.zeros(labels.shape)
    train_label[mask_train, :] = labels[mask_train, :]
    y_testOld[mask_testOld, :] = labels[mask_testOld, :]
    return adj, features,train_label, y_testArr, mask_train, mask_train1,mask_testArr, y_testOld, mask_testOld

def get_class_number(class_num, Y):
    num_ev_class = np.zeros((1, class_num), dtype=np.int)
    idx_ev_class = dict()
    for i in range(1, class_num + 1):
        # 通过这个我们可以查找到元素所在的位置，下标从0开始，Y==1,2,3,4,5,6
        tmp_idx = np.where(Y == i)[0]
        n_tmp_idx = len(tmp_idx)
        num_ev_class[0, i - 1] = n_tmp_idx
        key = str(i - 1)
        idx_ev_class[key] = tmp_idx
    # num_ev_class表示一个一行9列的向量，每个代表这个类有多少个元素
    # idx_ev_class是一个字典，每一个元素表示的是这个类所占的位置
    return num_ev_class, idx_ev_class


# 我们取1%的标记样本来进行数据的训练，其它样本作为测试集
def create_index(class_num, num_ev_class, idx_ev_class, train_rate):
    idx_tr = []  # list
    idx_te = []  # list

    for i in range(class_num):
        key1 = str(i)
        train_num = np.int(np.fix(num_ev_class[0, i] * train_rate))
        # 对原来的进行重新洗牌
        rnd_idx = np.random.permutation(num_ev_class[0, i])
        rnd_idx_tr = rnd_idx[range(train_num)]
        # 我们通过这种for循环遍历，得到每一类随机的1%的训练样本
        location_tr = idx_ev_class[key1][rnd_idx_tr]
        idx_tr.append(location_tr)

        # 出去我们的训练样本，剩余的就是我们的测试样本
        rnd_idx_te = rnd_idx[range(train_num, num_ev_class[0, i])]
        location_te = idx_ev_class[key1][rnd_idx_te]
        idx_te.append(location_te)

    # 在这里idx_tr表示的是一个列表，列表里面有6种元素，每一种元素为一个列表，列表里面的内容是这一类中
    # 这个元素所处的位置
    # 在这里idx_te表示的是一个列表，列表里面有6种元素，每一种元素为一个列表，列表里面的内容是这一类中
    # 这个元素所处的位置
    train_index = idx_tr[0]
    test_index = idx_te[0]
    # 通过下面的索引，将不同种类的元素，按顺序放入一个列表中
    for i in range(1, class_num):
        train_index = np.hstack((train_index, idx_tr[i]))
    for i in range(1, class_num):
        test_index = np.hstack((test_index, idx_te[i]))
    # 我们最终得到了训练集的索引和测试集的索引， 索引为标签数据
    return train_index, test_index

def preprocess_adj(adj):
    """对邻接矩阵做一个处理,最后返回了一三个元素，坐标，值，形状"""
    return sparse_to_tuple(adj)


def normalize_adj(adj):
    """对邻接矩阵做规范化处理."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 对特征进行规范化的操作
def preprocess_features(features):
    # 对特征进行行规范化
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


# 最终我们得到的是一个稀疏矩阵
def sparse_to_tuple(sparse_mx):
    """最后返回三个值，分别代表值的坐标位置，值的大小，形状"""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = sp.lil_matrix(mx)
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# 构造一个字典来进行，模型全局变量初始化模型的参数添加，即placeholders中的占位值赋予初始值
def construct_feed_dict(features, support,labels, mask_lables, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: mask_lables})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

