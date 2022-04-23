# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:38:56 2019

@author: Chenaniah
"""
import data_prepare
from conherent_matrix import ConherntMatrix
import utils
import graph
import time

if __name__ == "__main__":
    T, label, V, Fea_File = utils.loadData()
    print("End of loading for data")

    preData = data_prepare.Data(Fea_File, label)
    features, index = preData.mainDeal()
    print("End of preparing for data")

    conherentMatrix = ConherntMatrix(V, label, T)
    Cmat = conherentMatrix.mainCall()
    print("Complete for cmat to vmat")

    graph = graph.GraphNetwoks()
    startTime = time.time()
    graph.KNN_graph(Cmat, features, index, 30)
    # graph.KNN_graphNoSpatial(Cmat, features, index, 30)
    # graph.KNN_graphNoFuzzy(Cmat, features, index, 30)
    print("End for construct graph")
    endTime = time.time()
    print("the time of constructing graph", endTime - startTime)
