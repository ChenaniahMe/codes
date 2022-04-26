import math
from matplotlib.pyplot import axis
from networkx.algorithms.operators.product import _init_product_graph
from numpy.core.numeric import outer
import torch
import torch.nn as nn
from utils import *

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import sys
import numpy as np
import config
import pandas as pd

class PerLayer(Module):
    def __init__(self, in_features, out_features, queues,dropout, act=F.relu):
        super(PerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act # should tune whether F.relu or F.tanh
        self.dropout = nn.Dropout(dropout)
        self.layer = torch.nn.Linear(self.in_features, self.out_features)
        # self.reset_parameters()

    def reset_parameters(self):
        """Glorot & Bengio (AISTATS 2010) init."""
        stdv = math.sqrt(6.0 / (self.weight[0].size(0) + self.weight[0].size(1)))
        self.weight[0].data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = self.layer(input)
        output = torch.mm(adj, output)
        return self.act(output)
