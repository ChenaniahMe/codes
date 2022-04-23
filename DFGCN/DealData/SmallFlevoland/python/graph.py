# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:13:20 2018

@author: Eason
"""
import numpy as np
import scipy.sparse as sp
import os
import scipy.io as scio
import time
from multiprocessing import Pool
import config


class GraphNetwoks(object):
    def __init__(self):
        self.B = 0  # is a changing value

    def comDis(self, A):
        '''compute Wishart distance
        Attributes:
            A:
                is Convariance in map using parallel
        '''
        return abs(np.log(np.linalg.det(self.B) / np.linalg.det(A)) + np.trace(np.linalg.inv(self.B).dot(A))) - 3

    def KNN_graph(self, Cmat, feature, index, K):
        ''' KNN to construct adj in GCN
        Attributes:
            Cmat:
                is a convariance matrix with PolSAR
            feature:
                is 41 numbers features by fetature extration.
            index:
                data index
            K:
                K neighboor numbder in KNN
        '''
        N = feature.shape[0]
        X_2 = feature
        r_2, c_2 = index // config.corrA, index % config.corrB
        adj = sp.lil_matrix((N, N))
        dist = np.zeros((N, K))
        neighborhood = np.zeros((N, K), dtype=np.int64)
        #        A=Cmat[:]
        for i in range(N):
            print("Search the neighbors for the %d sample." % i)
            X_1 = feature[i, :]
            self.B = Cmat[i]
            #            with Pool(config.CORE) as p:
            #                a=p.map(self.comDis, A)
            # get coordinate distance distance_C
            r_1, c_1 = index[i] // config.corrA, index[i] % config.corrB
            distance_C = np.max(np.vstack((np.abs(r_2 - r_1), np.abs(c_2 - c_1))), axis=0)

            # get tow distances, Distances_E for 41 features distances
            # distance_W for wishart distance
            global distance_E
            distance_E = np.sum((X_2 - X_1) ** 2, axis=1)

            #            global distance_W
            #            distance_W=a

            # normalize two distances
            distance_E = (distance_E - min(distance_E)) / (max(distance_E) - min(distance_E))
            #            distance_W = (distance_W-min(distance_W))/(max(distance_W)+min(distance_W))
            # assign weight
            #            distance_R=0.5*distance_E + 0.5*distance_W
            distance_R = distance_E
            distance = distance_C * distance_R
            idx = np.argsort(distance)
            neighborhood[i, :] = idx[1:K + 1]
            dist[i, :] = distance[idx[1:K + 1]]

        for i in range(N):
            print("Compute the similarity for the %d sample and its neighbors." % i)
            delta1 = np.sqrt(dist[i, K - 1])
            for j in range(K):
                delta2 = np.sqrt(dist[neighborhood[i, j], K - 1])
                adj[i, neighborhood[i, j]] = np.exp(-dist[i, j] / (delta1 * delta2))
        adj = sp.coo_matrix(adj)
        sp.save_npz(os.path.join(config.adj_name), adj)
        return adj

    def KNN_graphNoSpatial(self, Cmat, feature, index, K):
        ''' KNN to construct adj in GCN
        Attributes:
            Cmat:
                is a convariance matrix with PolSAR
            feature:
                is 41 numbers features by fetature extration.
            index:
                data index
            K:
                K neighboor numbder in KNN
        '''
        N = feature.shape[0]
        X_2 = feature
        adj = sp.lil_matrix((N, N))
        dist = np.zeros((N, K))
        neighborhood = np.zeros((N, K), dtype=np.int64)
        for i in range(N):
            print("Search the neighbors for the %d sample." % i)
            X_1 = feature[i, :]
            # get tow distances, Distances_E for 41 features distances
            # distance_W for wishart distance
            global distance_E
            distance_E = np.sum((X_2 - X_1) ** 2, axis=1)

            # normalize distances
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
        adj = sp.coo_matrix(adj)
        sp.save_npz(os.path.join("adj.npz"), adj)
        return adj

    def KNN_graphNoFuzzy(self, Cmat, feature, index, K):
        ''' KNN to construct adj in GCN
        Attributes:
            Cmat:
                is a convariance matrix with PolSAR
            feature:
                is 41 numbers features by fetature extration.
            index:
                data index
            K:
                K neighboor numbder in KNN
        '''
        N = feature.shape[0]
        X_2 = feature
        adj = sp.lil_matrix((N, N))
        dist = np.zeros((N, K))
        neighborhood = np.zeros((N, K), dtype=np.int64)
        for i in range(N):
            print("Search the neighbors for the %d sample." % i)
            X_1 = feature[i, :]
            # get tow distances, Distances_E for 41 features distances
            # distance_W for wishart distance
            global distance_E
            distance_E = np.sum((X_2 - X_1) ** 2, axis=1)

            # normalize distances
            distance_E = (distance_E - min(distance_E)) / (max(distance_E) - min(distance_E))

            distance = distance_E
            idx = np.argsort(distance)
            neighborhood[i, :] = idx[1:K + 1]
            dist[i, :] = distance[idx[1:K + 1]]

        for i in range(N):
            print("Compute the similarity for the %d sample and its neighbors." % i)
            for j in range(K):
                adj[i, j] = dist[i, j]
        adj = sp.coo_matrix(adj)
        sp.save_npz(os.path.join("adj.npz"), adj)
        return adj

# if __name__=="__main__":
#    startTime = time.time()
#    Cmat=scio.loadmat("C.mat")
#    Cmat=Cmat['C']
#    features = np.load('Feature.npy')
#    index = np.load('index.npy')
#    constrctGraph = GraphNetwoks()
#    constrctGraph.KNN_graph(Cmat, features, index, 30)
#    endTime = time.time()
#    print("the time of constructing graph", endTime-startTime)
