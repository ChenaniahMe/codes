import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os
import torch
import copy
import config
import random
def masked_softmax_cross_entropy(logits, labels, mask):
    # print("logits.shape",logits.shape,"labels.shape",labels.shape,"mask.shape",mask.shape)
    #使用softmax_cross_entropy_with_logits，自己实现
    # print("logits",torch.mean(logits))
    y = torch.softmax(logits,dim=1)
    tf_log = torch.log(y)
    pixel_wise_mult = labels*tf_log
    loss = -pixel_wise_mult
    loss = torch.sum(loss,axis=1)
    # print("torch.sum(loss)",torch.sum(loss))
    # print("mask1",mask)
    # print("loss",loss)
    mask = mask / torch.mean(mask)
    # print("mask2",mask)
    loss = loss * mask
    return torch.mean(loss)

def masked_softmax_cross_entropyb(logits, labels, mask):
    # print("logits.shape",logits.shape,"labels.shape",labels.shape,"mask.shape",mask.shape)
    #使用softmax_cross_entropy_with_logits，自己实现
    # print("logits",torch.mean(logits))
    y = torch.softmax(logits,dim=1)
    tf_log = torch.log(y)
    pixel_wise_mult = labels*tf_log
    loss = -pixel_wise_mult
    loss = torch.sum(loss,axis=1)
    # print("torch.sum(loss)",torch.sum(loss))
    # print("mask1",mask)
    # print("loss",loss)
    mask = mask / torch.mean(mask)
    # print("mask2",mask)
    loss = loss * mask
    return torch.mean(loss)

def masked_accuracy(logits, labels, mask):
    accuracy_all = torch.argmax(logits, 1).eq(torch.argmax(labels, 1)).float()
    # print("torch.argmax(logits, 1)",torch.argmax(logits, 1))
    # print("torch.argmax(labels, 1))",torch.argmax(labels, 1))
    # print("accuracy_all",accuracy_all)
    mask /= torch.mean(mask)
    accuracy_all *= mask
    return torch.mean(accuracy_all)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        print("citeseer"*10)
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print("labels",labels)
    idx_test = test_idx_range.tolist()
    # print("idx_test",idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])


    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # print("y_test")
    # for i in range(len(y_test)):
    #     print(y_test[i])
    return adj, labels,features, y_train, y_val, y_test, train_mask, val_mask, test_mask,val_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
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
    print(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # print("features",features)
    return sparse_to_tuple(features)



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

# Accuracy
def accuracy(output, labels,idx):
    preds = output.max(1)[1].type_as(labels)
    # print("output",output)
    # print("preds",preds)
    # print("labels",labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # print("len(labels)",len(labels))
    return correct / len(labels)

def make_sq_mul(kernel):
    dim = kernel.size(1)//4
    # print("dim",dim)
    # print("kernel.size()",kernel.size())

    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton



def make_quaternion_mul(kernel):
    #对行的维度做了扩展
    # print("kernel.shape",kernel.shape)
    dim = kernel.size(1)//4
    # print("dim",dim)
    # print("kernel.size()",kernel.size())

    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    # print("hamilton.shape",hamilton.shape)
    return hamilton

def perwmul(kernel):
    dim = kernel.size(1)//4
    print("kernel.shape",kernel.shape)
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, i, j, k], dim=1)  # 0, 1, 2, 3
    i2 = torch.cat([r, i, j, k], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([r, i, j, k], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([r, i, j, k], dim=1)  # 3, 2, 1, 0

    resmul = torch.cat([r2, i2, j2, k2], dim=0)
    assert kernel.size(1) == resmul.size(1)
    # print("hamilton.shape",hamilton.shape)
    return resmul

def quaternion_preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = features.todense()
    # #old
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    print("features",features)
    #new
    return torch.from_numpy(features).float()
    # return torch.from_numpy(features)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_bias(adj, sizes, nhood=1):  
    adj = adj.todense()
    # oadj = torch.Tensor(copy.deepcopy(adj)).cuda()
    adj = adj[np.newaxis]

    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    # print("adj.shape",adj.shape)
    # print("mt.shape",mt.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    adj = torch.Tensor(-1e9 * (1.0 - mt))[0].cuda()
    # adj = torch.add(adj, oadj)
    adj = torch.softmax(adj,dim=1)
    return adj


def featuresw(features, weights,splitn):
    
    res_features =[]
    for ta in torch.split(features,features.shape[1]//4,dim=1):
        res_features.append(torch.split(ta,features.shape[0]//4,dim=0))

    res_weights =[]
    for tb in torch.split(weights,weights.shape[1]//4,dim=1):
        res_weights.append(torch.split(tb,weights.shape[0]//4,dim=0))

    res = []
    for i in range(len(res_features)):
        tres = []
        for j in range(len(res_weights[i])):
            tres.append(res_features[i][j]@res_weights[i][j])
        res.append(torch.cat(tres))     
    return torch.cat(res,axis=1)

def featureswq(features, weights,splitn):
    # print("features.shapes",features.shape,"weights.shapes",weights.shape)
    # return features@weights

    res_features =[features]
    res_weights =[]
    for tb in torch.split(weights,1,dim=1):
        res_weights.append(tb)
    # print("len(res_weights)",len(res_weights))
    random.shuffle(res_weights[:4]) #此处有问题，貌似并没有发生变化
    res = []
    for i in range(len(res_features)):
        tres = []
        for j in range(len(res_weights)):
            # print("res_weights[j]",res_weights[j],j)
            # print("type(res_features[i])",torch.sum(res_features[i]))
            # res_features[i] = torch.tensor(res_features[i], dtype=torch.float64).cuda()
           
            # print("res_weights[j])",res_weights[j])
            tres.append(res_features[i]@res_weights[j])
        nres = [tres[0]+tres[1],tres[0]-tres[1],tres[2],tres[3]]
        res.append(torch.cat(nres,axis=1))
    return torch.cat(res,axis=0)
    
#### 扩散图a
def A_to_diffusion_kernel(A, k):
    """
    Computes [A**0, A**1, ..., A**k]
    :param A: 2d numpy array
    :param k: integer, degree of series
    :return: 3d numpy array [A**0, A**1, ..., A**k]
    """
    A = A.todense()
    assert k >= 0
    Apow = [np.identity(A.shape[0])] #矩阵的行数和列数
    if k > 0: #扩散阶数而得到A不同
        d = A.sum(0)
        Apow.append(A / (d + 1.0))
        for i in range(2, k + 1):
            Apow.append(np.dot(A / (d + 1.0), Apow[-1]))
    
    print("Apow",Apow)
    return torch.Tensor(np.transpose(np.asarray(Apow, dtype='float32'), (1, 0, 2))).cuda()

#### 扩散图b
def A_to_diffusion_kernel(A, k):
    """
    Computes [A**0, A**1, ..., A**k]
    :param A: 2d numpy array
    :param k: integer, degree of series
    :return: 3d numpy array [A**0, A**1, ..., A**k]
    """
    A = A.todense()
    assert k >= 0
    Apow = [np.identity(A.shape[0])] #矩阵的行数和列数
    if k > 0: #扩散阶数而得到A不同
        d = A.sum(0)
        Apow.append(A / (d + 1.0))
        for i in range(2, k + 1):
            Apow.append(np.dot(A / (d + 1.0), Apow[-1]))
    
    return torch.Tensor(np.array(Apow[-1])).cuda()