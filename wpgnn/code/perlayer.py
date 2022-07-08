from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor, remainder
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
from util import *
import config
from parse import parse_args
args = parse_args()

def get_inc(edge_index):
    # compute the incident matrix
    size = edge_index.sizes()[1]
    
    row_index = edge_index.storage.row()
    col_index = edge_index.storage.col()
    
    #What happens if duplicate edges are not deleted
    mask = row_index >= col_index # remove duplicate edge and self loop
    row_index = row_index[mask]
    col_index = col_index[mask]

    #.numel() Returns the number of elements in the array
    edge_num = row_index.numel()
    row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
    col = torch.cat([row_index, col_index])
    value = torch.cat([torch.ones(edge_num), -1*torch.ones(edge_num)]).cuda()
    # print("size",size)
    # print("edge_num",edge_num)
    inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                        sparse_sizes=(edge_num, size))

    #Modify
    # row = torch.cat([torch.arange(edge_num)]).cuda()
    # col = torch.cat([row_index])
    # value = torch.cat([torch.ones(edge_num)]).cuda()
    # inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
    #                 sparse_sizes=(edge_num, size))
    # print("inc.to_dense().shape",inc.to_dense().shape)
    # print("inc.to_dense()",inc.to_dense())
    # print("inc",inc)
    return inc

def inc_norm(inc, edge_index):
    ## edge_index: unnormalized adjacent matrix
    ## normalize the incident matrix
    edge_index = torch_sparse.fill_diag(edge_index, 1.0) ## add self loop to avoid 0 degree node
    deg = torch_sparse.sum(edge_index, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    inc = torch_sparse.mul(inc, deg_inv_sqrt.view(1, -1)) ## col-wise
    return inc

def check_inc(edge_index, inc):
    nnz = edge_index.nnz()
    deg = torch.eye(edge_index.sizes()[0]).cuda()
    adj = edge_index.to_dense()
    lap = (inc.t() @ inc).to_dense()
    lap2 = deg - adj
    diff = torch.sum(torch.abs(lap2-lap)) / nnz
    # assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'

class PerLayer(torch.nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                queues,
                 K: int, 
                 dropout: float = 0,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):

        super(PerLayer, self).__init__()
        self.K = K
        self.queues = queues
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        if self.queues == 1:
            interres = self.out_features // args.block_num
            dim_main = [interres]*(args.block_num-1)
            indexdim =  dim_main+ [self.out_features-sum(dim_main)]

            self.weightf = torch.nn.ParameterList([Parameter(torch.FloatTensor(self.in_features, indexdim[i])) for i in range(args.block_num)])
            # print("pernet.shape",)
            for i in range(len(self.weightf)):
                self.weightf[i] = Parameter(self.weightf[i])
            self.reset_parametersf()

        elif self.queues == 2:
            print("self.in_features",self.in_features)
            interres = self.out_features // args.block_num
            dim_main = [interres]*(args.block_num-1)
            indexdim =  dim_main+ [self.out_features-sum(dim_main)]
            
            self.weights = torch.nn.ParameterList([Parameter(torch.FloatTensor(self.in_features+args.extend_dim, indexdim[i])) for i in range(args.block_num)])
            for i in range(len(self.weights)):
                self.weights[i] = Parameter(self.weights[i])
            self.reset_parameterss()

    def reset_parametersf(self):
        import math
        for i in range(len(self.weightf)):
            stdv = math.sqrt(6.0 / (self.weightf[i].size(0) + self.weightf[i].size(1)))
            self.weightf[i].data.uniform_(-stdv, stdv)

    def reset_parameterss(self):
        import math
        for i in range(len(self.weights)):
            stdv = math.sqrt(6.0 / (self.weights[i].size(0) + self.weights[i].size(1)))
            self.weights[i].data.uniform_(-stdv, stdv)
    # def reset_parameters(self):
    #     pass

    def forward(self,x: Tensor, 
                edge_index: Adj, 
                edge_weight: OptTensor = None, 
                data=None) -> Tensor:
        inc_mat = get_inc(edge_index=edge_index)
        inc_mat = inc_norm(inc=inc_mat, edge_index=edge_index)
        self.init_z = torch.zeros((inc_mat.sizes()[0], x.size()[-1])).cuda()
        
        edge_index = gcn_norm(
                    edge_index, edge_weight, False,
                    add_self_loops=self.add_self_loops, dtype=x.dtype)

        #Weight-perceputal convolution
        if self.queues == 1:
            outputs = []
            for i in range(len(self.weightf)):
                tx = featureswq(x,self.weightf[i],1)
                outputs.append(tx)
            x = torch.cat(outputs,axis=1)

        #Weight-perceputal convolution
        elif self.queues == 2:
            outputs = []
            # print("self.weightf[0].shape",self.weightf[0].shape)
            for i in range(len(self.weights)):
                tx = featureswq(x,self.weights[i],2)
                outputs.append(tx)
            x = torch.cat(outputs,axis=1)
            
        else:
            hh = x
            # for k in range(self.K):

            #     # print("hh.shape", hh.shape,"edge_index.sizes()", edge_index.sizes())
            #     x =  0.25 *hh + (0.85) * (edge_index@x)
                
            for k in range(self.K):
                y =  0.25 *hh + (1-0.25) * (edge_index@x)
                x_bar = y - 0.25 * (inc_mat.t() @ self.init_z)
                z_bar  = self.init_z + 2.0 * (inc_mat @ x_bar)
                self.init_z  = self.L21_projection(z_bar, lambda_=3)
                x = y - 0.25 * (inc_mat.t() @ self.init_z)
        return x

    def L21_projection(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]
        return scale.unsqueeze(1) * x