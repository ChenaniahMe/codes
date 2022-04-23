# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:54:19 2019

@author: Chenaniah
"""
import numpy as np
from scipy.io import loadmat
import h5py

def loadData():
    Fea_File = loadmat("Fea_V.mat")
    T = loadmat("T.mat")['T']
    label = loadmat("label.mat")['label']
    V=1/(np.power(2,1/2))* np.array([[ 1.        ,  0.        ,  1.        ],
       [ 1.        ,  0.        , -1.        ],
       [ 0.        , np.power(2,1/2) ,  0.        ]])
#    features = np.load('Feature.npy')
#    index = np.load('index.npy')
    return T, label, V, Fea_File
