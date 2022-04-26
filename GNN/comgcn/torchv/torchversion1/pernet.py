import torch
from perlayer import PerLayer
import torch.nn.functional as F

class PerNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(PerNet, self).__init__()
        self.layera = PerLayer(nfeat, 32, dropout=dropout,act=F.relu, queues = 1)  # should tune whether relu or tanh
        # self.q4gnn2 = QGNNLayer(nhid, nclass, dropout=dropout, quaternion_ff=False, act=lambda x:x) # quaternion_ff=False --> QGNN becomes GCN
        self.layerb = PerLayer(32, nclass, dropout=dropout, act=lambda x:x,queues=2)
    def forward(self, x, adj):
        x1 = self.layera(x, adj)
        # print(adj.shape)
        x2 = self.layerb(x1, adj)
        return x2