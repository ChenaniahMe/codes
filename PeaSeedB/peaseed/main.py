import imp
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from data import all_dataset
from data import all_labels

batch=50
iteration=300

#数据载入

#模块搭建
class ResBlock(torch.nn.Module):
    def __init__(self,channels_in):
        super().__init__()
        self.conv1=torch.nn.Conv2d(channels_in,30,5,padding=2)
        self.conv2=torch.nn.Conv2d(30,channels_in,3,padding=1)

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        return F.relu(out+x)

#网络搭建
class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,20,5)
        self.conv2=torch.nn.Conv2d(20,15,3)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.resblock1=ResBlock(channels_in=20)
        self.resblock2=ResBlock(channels_in=15)
        self.full_c=torch.nn.Linear(375,10)

    def forward(self,x):
        size=x.shape[0]
        x=F.relu(self.maxpool(self.conv1(x)))
        x=self.resblock1(x)
        x=F.relu(self.maxpool(self.conv2(x)))
        x=self.resblock2(x)
        x=x.view(size,-1)
        x=self.full_c(x)
        return x

#损失函数、优化器、学习率衰减
model=ResNet()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
schedular=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.999)

#训练函数
def train():
    for epoch in range(iteration):
        l=0.0
        # train_data,train_labels=data
        # print("train_data.shape",train_data.shape)
        # print("train_labels.shape",train_labels.shape)
        train_data = all_dataset
        train_labels = all_labels.long()
        # print("train_data.shape",train_data.shape)
        # print("train_labels.shape",train_labels.shape)
        # print("train_data[0]",train_data[0][0])
        import sys
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14,8))
        # ax1 = fig.add_subplot(1,1,1)
        arrs = np.array(train_data[0][0].tolist())
        # ax1.imshow(arrs,cmap='binary')
        # fig.savefig("411.png")
        # print("arrs.shape",arrs.shape)
        im = Image.fromarray(arrs)
        # im.show()
        # im.save("./1.png")
        # print("True")
        # sys.exit()
        # print("train_labels",train_labels)
        optimizer.zero_grad()
        pred_data=model(train_data)
        loss=criterion(pred_data,train_labels)
        loss.backward()
        l+=loss.item()
        optimizer.step()
        schedular.step()
        print("epoch:",epoch,"loss:",l)

#测试函数
def test():
    with torch.no_grad():
        correct=0.0
        total=0.0
        # test_data,test_labels=data
        test_data = all_dataset
        test_labels = all_labels.long()
        pred_data=model(test_data)
        _,pred_labels=torch.max(pred_data,dim=1)
        total+=test_labels.shape[0]
        correct+=(pred_labels==test_labels).sum().item()
        print("准确率为：",correct*100.0/total,"%")

#主函数
if __name__ == '__main__':
    train()
    test()

