# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:52:23 2019
@author: Chenaniah
"""

import numpy as np 
from scipy.io import loadmat, savemat
from multiprocessing import Pool
import utils
import config
class ConherntMatrix(object):
    ''' T(coherent matrix) converted into C(Covariance matrix)
    Attributes:
        V: is a matrix
        V=1/(np.power(2,1/2))*np.array([[ 1.        ,  0.        ,  1.        ],
           [ 1.        ,  0.        , -1.        ],
           [ 0.        ,  1.41421356,  0.        ]])
        Tmat: is a conherent matrix from raw data
        Cov=np.linalg.inv(V).dot(T).dot(V)
    '''
    def __init__(self, V, label, T):
        self.V=V
        self.label = label
        self.T = T
        self.A = 0
        
    def transToCov(self, A):
        result = np.linalg.inv(self.V).dot(A).dot(self.V)
        return result
    
    def mainCall(self):
        '''the function call order
        Args:
            V: is a matrix from initialzation
        '''
        A = self.prepareData(self.label)[:,0]
        with Pool(config.CORE) as p: 
            Cov=p.map(self.transToCov, A)
        return Cov
    
    def prepareData(self, label):
        '''get T of possessing label
        Args:
            lable: is a matrix from label data
        '''
        label = self.label.reshape(1,label.shape[0]*label.shape[1])[0]
        index = np.where(label != 0)[0]
        self.T=self.T.reshape(self.T.shape[0]*self.T.shape[1],1)[index,:]
        return self.T
        
# if __name__=="__main__":
#     T, label, V, features, index = utils.loadData()
#     print("data加载完毕")
#     conherentMatrix = ConherntMatrix(V, label, T)
#     Cmat = conherentMatrix.mainCall()
#
#     savemat("C.mat",{'__header__':"b'MATLAB 5.0 MAT-file, \
#                           Platform: PCWIN64, Created on: Sat Apr 15 13:34:56 2017'", '__version__':"1.0", '__globals__':[], 'C':Cmat})