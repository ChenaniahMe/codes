import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import os
def KNN_graph(feature, K):
    feature=np.array(feature)
    N = feature.shape[0]
    X_2 = feature
    adj = sp.lil_matrix((N, N))
    dist = np.zeros((N, K))
    neighborhood = np.zeros((N, K), dtype=np.int64)
    #        A=Cmat[:]
    for i in range(N):
        print("Search the neighbors for the %d sample." % i)
        X_1 = feature[i, :]
        # get tow distances, Distances_E for 41 features distances
        # distance_W for wishart distance
        global distance_E
        distance_E = np.sum((X_2 - X_1) ** 2, axis=1)
        # normalize two distances
        distance_E = (distance_E - min(distance_E)) / (max(distance_E) - min(distance_E))
        distance = distance_E
        idx = np.argsort(distance)
        neighborhood[i, :] = idx[1:K + 1]
        dist[i, :] = distance[idx[1:K + 1]]

    for i in range(N):
        print("Compute the similarity for the %d sample and its neighbors." % i)
        delta1 = np.sqrt(dist[i, K - 1])
        for j in range(K):
            delta2 = np.sqrt(dist[neighborhood[i, j], K - 1])
            adj[i, neighborhood[i, j]] = np.exp(-dist[i, j] / (delta1 * delta2))
    adj = sp.coo_matrix(adj).todense() + identity(N).toarray()
    adj = np.array(adj)
    result = pd.DataFrame(np.matmul(adj,feature),columns=[['outdoorTemp', 'outdoorHum', 'outdoorAtmo','indoorHum', 'indoorAtmo']])
    return result
def KNNTwo(feature, K):
    hy_dis = cdist(feature, feature, metric='euclidean')
    hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
    hy_dis = np.power(hy_dis,2)
    idx_k_pos = np.argsort(hy_dis)[:, :K]
    idx_len_pos = idx_k_pos.shape[0]
    hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
    hy_i_k_pos = hy_k_pos[:, -1]
    hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
    mole_pos = -hy_k_pos
    deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
    hy_ij_pos = np.exp(mole_pos / deno_pos)
    A_ij_pos = np.zeros((hy_ij_pos.shape[0], hy_ij_pos.shape[0]))
    A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos
    hy_matmul_pos = np.matmul(A_ij_pos, feature)
    return hy_matmul_pos

def KNNThree(feature, K):
    hy_dis = cdist(feature, feature, metric='euclidean')
    hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
    idx_k_pos = np.argsort(hy_dis)[:, :K]
    idx_len_pos = idx_k_pos.shape[0]
    hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
    hy_i_k_pos = hy_k_pos[:, -1]
    hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
    mole_pos = -hy_k_pos
    deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
    hy_ij_pos = np.exp(mole_pos / deno_pos)
    A_ij_pos = np.zeros((hy_ij_pos.shape[0], hy_ij_pos.shape[0]))
    A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos
    hy_matmul_pos = np.matmul(A_ij_pos, feature)
    return hy_matmul_pos
def deal_outliers(df, pro_name):
    Q1 = df[pro_name].quantile(0.25)
    Q3 = df[pro_name].quantile(0.75)
    IQR = Q3 - Q1
#    df = df[~( (df[pro_name] < (Q1-1.5*IQR)) | (df[pro_name] > (Q3+1.5*IQR)) )]
    df[(df[pro_name] < (Q1-1.5*IQR)) | (df[pro_name] > (Q3+1.5*IQR))]=np.nan
    return df

def insert_value(data, method):
    if method=='linear':
        data =  data.interpolate()
    if method=='ffill':
        data = data.fillna(method='ffill')
    if method == 'poly2':
        data = data.interpolate(method='polynomial', order=2)
    if method == 'poly3':
        data = data.interpolate(method='polynomial', order=3)
    return data

def deal_outliers(data, col_name):
    data = data[col_name]
    if col_name=='outdoorAtmo':
        data[data < 955] = np.nan
        data[data > 1000] = np.nan
    if col_name == 'indoorAtmo':
        data[data < 600] = np.nan
    return data