import time
import numpy as np

from gat import GAT
from utils import process
import torch.optim as optim
import torch

# checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training params
batch_size = 1
#nb_epochs = 100000
#modify epochs
nb_epochs = 150
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [10, 1] # additional entry for the output layer
residual = False
nonlinearity = torch.relu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]
print("nb_classes",nb_classes)
adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = torch.Tensor(y_train[np.newaxis]).to(device)
y_val = y_val[np.newaxis]
y_test = torch.Tensor(y_test[np.newaxis]).to(device)
train_mask = torch.Tensor(train_mask[np.newaxis]).to(device)
val_mask = val_mask[np.newaxis]
test_mask = torch.Tensor(test_mask[np.newaxis]).to(device)

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)
# print(biases)

ftr_in =torch.Tensor(features).to(device)
bias_in = torch.Tensor(biases).to(device)
is_train = True
attn_drop = 0.6
ffd_drop = 0.6

print("ftr_in.shape[2],hid_units[0]",ftr_in.shape[2],hid_units[0])
m = model(ftr_in.shape[2],hid_units[0],n_heads,nb_classes,attn_drop,ffd_drop).to(device)
# print("m",m)
# print("m.parameters()",m.parameters())
optimizer = optim.Adam(m.parameters(),lr=lr)
print("model",m)
for epoch in range(nb_epochs):
    logits = m(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)     
    # print("logits",logits.shape)      
    log_resh = torch.reshape(logits, [-1, nb_classes])
    lab_resh = torch.reshape(y_train, [-1, nb_classes])
    msk_resh = torch.reshape(train_mask, [-1])
    # print("m.named_parameters",m.named_parameters)
    loss = m.masked_softmax_cross_entropy(log_resh,lab_resh,msk_resh)
    acc = m.masked_accuracy(log_resh,lab_resh,msk_resh)
    print("epoch:",epoch,"loss",loss,"acc",acc) 
    loss.backward(retain_graph=True)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()       # backpropagation, compute gradients
    optimizer.step()
    if epoch==nb_epochs-1:
        lab_resh = torch.reshape(y_test, [-1, nb_classes])
        msk_resh = torch.reshape(test_mask, [-1])
        acc = m.masked_accuracy(log_resh,lab_resh,msk_resh)
        print(acc)