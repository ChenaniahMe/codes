from __future__ import division
from __future__ import print_function

import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import config
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from utils import *
from pernet import PerNet
# torch.set_default_dtype(torch.float64)
if config.israndom:
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================
parser = ArgumentParser("PerNet", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset", default= "citeseer", help="Name of the dataset.")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden_size//4 = number of quaternion units within each hidden layer.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
args = parser.parse_args()

# Load data
# adj,labels , features, y_train, y_val, y_test, train_mask, val_mask, test_mask,percep_mask = load_data(args.dataset)
adj, labels, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, percep_mask = load_data(args.dataset)


labels = torch.from_numpy(labels).to(device)
# labels = torch.where(labels==1)[1]
idx_train = torch.where(torch.from_numpy(train_mask)==True)
idx_val = torch.where(torch.from_numpy(val_mask)==True)
idx_percep = torch.where(torch.from_numpy(percep_mask)==True)
idx_test = torch.where(torch.from_numpy(test_mask)==True)
# print("idx_test",idx_test)
# print("idx_test[0].shape",idx_test[0].shape)

#处理特征，最终的目标是具有多特征的形式
features = preprocess_features(features)
features = torch.sparse_coo_tensor(features[0].T,features[1],features[2]).float()
# print("features",features)
features = features.to_dense().cuda()
# features = torch.from_numpy(features).float().to(device)
# print("features",features,"features.shape",features.shape,"features.dtype",features.dtype )
#对adj作为bias的处理
madj = False
#对adj进行图扩散
difadj = False
#tensorflow对比实验
tadj = False
if madj:
    # adj = normalize_adj(adj + sp.eye(adj.shape[0])).tocoo()
    adj = adj_to_bias(adj, [features.shape[0]],nhood=1)
    # print("adj.shape",adj.shape) #(2708,2708)
    # print(adj)
elif difadj:
    adj = A_to_diffusion_kernel(adj, 0)
    print("adj.shape",adj.shape) #(2708,2708)
elif tadj:
    adj = preprocess_adj(adj)
    adj = torch.sparse_coo_tensor(adj[0].T,adj[1],adj[2]).float().to(device)
else:
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).tocoo()
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

# print("idx_test",idx_test)
# print("adj",adj.to_dense())
# print("y_train",y_train)
# print("y_test",y_test)
# Model and optimizer
train_mask = torch.Tensor(train_mask).to(device)
val_mask = torch.Tensor(val_mask).to(device)
test_mask = torch.Tensor(test_mask).to(device)
model = PerNet(nfeat=features.size(1), nhid=args.hidden_size, nclass=y_train.shape[1], dropout=args.dropout).to(device)
print("model",model)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

print("labels",labels)
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train],idx_train)

    loss_train = masked_softmax_cross_entropyb(output, labels, train_mask)
    acc_train = masked_accuracy(output,labels, train_mask)
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately, deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val],idx_val)

    loss_val = masked_softmax_cross_entropyb(output,labels, val_mask)
    acc_val = masked_accuracy(output,labels, val_mask)

    # print("test accuracy",accuracy(output[idx_test], labels[idx_test],idx_test))
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    # print("labels[idx_test]",labels[idx_test])
    loss_test = masked_softmax_cross_entropyb(output,labels, test_mask)
    acc_test = masked_accuracy(output,labels, test_mask)
    
    # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test],idx_test)
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    res = "{:.4f}".format(acc_test.item())
    f = open(config.resfile, "a+")
    f.write(res)
    f.write("\n")

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    # test()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()