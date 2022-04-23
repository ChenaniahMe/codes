# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:18:03 2019

@author: Chenaniah
"""
import numpy as np
from sklearn.preprocessing import normalize

class Data(object):
    '''Data Deal for data a little
    Fea_File:
        41 dimensions features by matalba
    Label:
        data label
    '''
    def __init__(self, Fea_File, Label):
        self.Fea_File = Fea_File
        self.Label = Label
        
    def remove_useless_samples(self, features, labels):
        '''del useless data that empty data
        '''
        index = np.where(labels != 0)[0]
        feature = features[index, :]
        return feature, index
    
    def mainDeal(self):
        ''''obtation data for GNCN to train
        '''
        Feature = self.Fea_File['Fea_V'][:]
        Feature = Feature.T
        print("self.Label.T.shape[0]",self.Label.T.shape[0])
        # self.Label = np.reshape(self.Label.T, self.Label.T.shape[0]*self.Label.T.shape[1])
        self.Label = np.reshape(self.Label, self.Label.T.shape[0]*self.Label.T.shape[1])
        Features, Index = self.remove_useless_samples(Feature, self.Label)
        Features = normalize(Features, norm='l2', axis=0)   # normalization
        Labels = self.Label[Index]  
        np.save("Feature.npy", Features)
        np.save("label.npy", Labels)
        np.save("index.npy", Index)
        return Features, Index
        
    
