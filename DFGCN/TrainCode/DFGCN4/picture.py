# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:42:16 2020

@author: Chenaniah
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
from PIL import Image
import h5py
# colors of ground truth smallFlevoland
Colors = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0],
                   [0, 0, 255], [102, 102, 0], [205, 149, 12],
                   [255, 0, 255]], dtype=np.uint8)

#oberp
# Colors = np.array([[255, 255, 255], [0, 0, 255], [255, 0, 0],
#                    [0, 255, 255], [0, 255, 0 ], [255, 0, 255],
#                    [255, 0, 255]], dtype=np.uint8)
#flevoland
# Colors = np.array([[255, 255, 255],[255, 0, 0], [255, 128, 0], [171, 138, 80], [255, 255, 0], [183, 0, 255],
#          [191, 191, 255], [90, 11, 255], [191, 255, 191], [0, 252, 255], [128, 0, 0],
#          [255, 182, 229], [0, 255, 0],[0, 131, 74], [0, 0, 255], [255, 217, 157]], dtype=np.uint8)
draw_picture=1
file_result_mask="result_mask.csv"
file_result_pvalue="result_pvalue.csv"
file_labelmat="../../DealData/SmallFlevoland/matlab/label.mat"
file_save="image_fle.bmp"
#label of test data
result_mask=pd.read_csv(file_result_mask).iloc[:,1]
#the value of probability for output
result_pvalue=pd.read_csv(file_result_pvalue).iloc[:,1:]
#get max index from each column
result_new_label = np.array(result_pvalue).argmax(axis=1)
#make train as -1 so that can not be recognized
result_new_label[result_mask[:]==0]=-1

label_ori = loadmat(file_labelmat)['label']
image_row=label_ori.shape[0]
image_col=label_ori.shape[1]
label_ori=label_ori.flatten()

#red 255, 0, 0
#while 255, 255, 255
#green 0, 255, 0  
#blue 0, 0, 255
#pink 255, 0, 255
#yellow 255, 255, 0
#old_label

#get old image
image_list=[]
if draw_picture==0:
    file_save = "image_fle.bmp"
    for i in range(len(label_ori)):
        print(i)
        image_list.append(Colors[int(label_ori[i])])
else:
    file_save = "image_fle_t.bmp"
    label_dealIndex=0
    for i in range(len(label_ori)):
       print(i)
       if(label_ori[i]!=0):
           if result_new_label[label_dealIndex]==-1:
               image_list.append(Colors[int(label_ori[i])])
           else:
               image_list.append(Colors[result_new_label[label_dealIndex]+1])
           label_dealIndex+=1
       else:
           image_list.append(Colors[label_ori[i]])
#
image_arr=np.array(image_list)
image_arr=image_arr.reshape(image_row,image_col,3)
image_result = Image.fromarray(image_arr)
image_result.save(file_save)
#image=Image.open("./picture/label_imag.bmp")
#image_arr=np.array(image)