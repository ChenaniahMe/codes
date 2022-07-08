import torch
import torch.nn.functional as F
from torch.nn import Linear
import copy
import numpy as np
import config
from perlayer import PerLayer

class PerNetGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop_lin1,prop_lin2, prop2, **kwargs):
        super(PerNetGNN, self).__init__()
        # self.lin1 = Linear(1, 1)
        # self.lin2 = Linear(hidden_channels+config.extend_dim, out_channels)
        self.nbeta =  torch.nn.Parameter(torch.FloatTensor([3]))
        self.gamma = torch.nn.Parameter(torch.FloatTensor([0.5]))
        # self.learn_beta = torch.nn.Parameter(torch.FloatTensor([0])).cuda()
        self.dropout = dropout
        self.prop_lin1 = prop_lin1
        self.prop_lin2 = prop_lin2
        self.prop2 = prop2

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop_lin1(x, adj_t, data=data)
        x = F.relu(x)
        x = self.prop_lin2(x, adj_t, data=data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop2(x, adj_t, data=data)
        
        return F.log_softmax(x, dim=1), x

def get_model(args, dataset,data):
    Model = PerNetGNN
    prop_lin1 =  PerLayer(data.x.shape[1],args.hidden_channels,queues=1,K=args.K)
    prop_lin2 =  PerLayer(args.hidden_channels,dataset.num_classes,queues=2,K=args.K)
    prop2 =  PerLayer(args.hidden_channels,dataset.num_classes,queues=3,K=args.K)
    model = Model(in_channels=data.num_features,
                       hidden_channels=args.hidden_channels,
                       out_channels=dataset.num_classes,
                       dropout=args.dropout,
                       num_layers=args.num_layers,
                       prop_lin1=prop_lin1, prop_lin2=prop_lin2,prop2=prop2).cuda()

    return model