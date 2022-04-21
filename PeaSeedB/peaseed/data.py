# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:36:12 2019

@author: Chenaniah
"""

# 数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).resize((28,28))
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        print("len(self.imgs)",len(self.imgs))
        return len(self.imgs)

dataseta=FlameSet('../datapea/SL-8001/种脐_tr')
datasetb=FlameSet('../datapea/SL-8002/种脐_tr')
datasetc=FlameSet('../datapea/SL-8003/种脐_tr')
datasetd=FlameSet('../datapea/SL-8004/种脐_tr')

all_labels = torch.Tensor([1]*len(dataseta) + [2]*len(datasetb)+ [3]*len(datasetc)+ [4]*len(datasetd))
all_dataseta = torch.cat([dataseta[i] for i in range(len(dataseta))]).unsqueeze(1)
all_datasetb = torch.cat([datasetb[i] for i in range(len(datasetb))]).unsqueeze(1)
all_datasetc = torch.cat([datasetc[i] for i in range(len(datasetc))]).unsqueeze(1)
all_datasetd = torch.cat([datasetd[i] for i in range(len(datasetd))]).unsqueeze(1)
all_dataset = torch.cat([all_dataseta,all_datasetb,all_datasetc,all_datasetd],dim=0)

print("all_dataset.shape",all_dataset.shape)
print("all_dataseta.shape",all_dataseta.shape)
print("all_datasetb.shape",all_datasetb.shape)
print("all_datasetc.shape",all_datasetc.shape)
print("all_datasetd.shape",all_datasetd.shape)
print("all_labels.shape",all_labels.shape)
