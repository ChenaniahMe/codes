from networkx.algorithms.traversal.edgedfs import FORWARD
import numpy as np
import torch

from utils import layers
from base_gattn import BaseGAttN
#

class GAT(BaseGAttN):
    def __init__(self,seq,out_sz,n_heads,nb_classes,in_drop, coef_drop):
        super(GAT, self).__init__()
        self.aseqf = torch.nn.ModuleList([torch.nn.Linear(seq, out_sz) for _ in range(n_heads[0])])
        self.aseqf1 =torch.nn.ModuleList([torch.nn.Linear(out_sz, 1) for _ in range(n_heads[0])])
        self.aseqf2 = torch.nn.ModuleList([torch.nn.Linear(out_sz, 1) for _ in range(n_heads[0])])
        # self.abias = [torch.nn.Parameter(torch.Tensor(2708,1).requires_grad_()) for _ in range(n_heads[0])]
        # self.abias = torch.nn.Parameter(torch.Tensor(2708,1))    
        # self.abias = [torch.nn.Parameter(torch.nn.Embedding(2708,1).weight.data) for _ in range(n_heads[0])]
        self.abias = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(2708,1), mean=0, std=1)) for _ in range(n_heads[0])])
        self.indropa = torch.nn.ModuleList([torch.nn.Dropout(1.0 - in_drop)  for _ in range(n_heads[0])])
        self.coefdropa = torch.nn.ModuleList([torch.nn.Dropout(1.0 - coef_drop)  for _ in range(n_heads[0])])

        self.bseqf = torch.nn.ModuleList([torch.nn.Linear(out_sz*n_heads[0], nb_classes) for _ in range(n_heads[-1])])
        self.bseqf1 =torch.nn.ModuleList([torch.nn.Linear(nb_classes, 1) for _ in range(n_heads[-1])])
        self.bseqf2 = torch.nn.ModuleList([torch.nn.Linear(nb_classes, 1) for _ in range(n_heads[-1])])
        # self.bbias = [torch.nn.Parameter(torch.Tensor(2708,1)) for _ in range(n_heads[-1])]
        # self.bbias = torch.nn.Parameter(torch.Tensor(2708,1))
        # self.bbias = [torch.nn.Parameter(torch.nn.Embedding(2708,1).weight.data) for _ in range(n_heads[-1])]
        self.bbias = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(2708,1), mean=0, std=1)) for _ in range(n_heads[-1])])
        self.indropb = torch.nn.ModuleList([torch.nn.Dropout(1.0 - in_drop)  for _ in range(n_heads[-1])])
        self.coefdropb = torch.nn.ModuleList([torch.nn.Dropout(1.0 - coef_drop)  for _ in range(n_heads[-1])])
       
        # self.seqf = torch.nn.Linear(out_sz*n_heads[0], out_sz)

    def forward(self,inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=torch.nn.ELU, residual=False):
        # print("inputs.shape",inputs.shape)
        attns = []
        for i in range(n_heads[0]):
            attns.append(layers.attn_head(self.aseqf[i],self.aseqf1[i],self.aseqf2[i],self.abias[i],inputs,self.indropa[i],self.coefdropa[i], bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        # print("attns:",attns)
        
        h_1 = torch.cat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            print("True"*100,"over")
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(self.seqf,self.seqf1,self.seqf2,h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = torch.cat(attns, axis=-1)
        out = []
        # print("h_1.shape",h_1.shape)
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(self.bseqf[i],self.bseqf1[i],self.bseqf2[i],self.bbias[i],h_1, self.indropb[i],self.coefdropb[i],bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        tout = out[0]
        for j in range(1,len(out)):
            tout = torch.add(tout,out[j])
        # print("torch.Tensor(out).shape",torch.Tensor(out).shape)
        # tvalue = torch.add(torch.Tensor(out))
        # print("tvalue",tvalue)
        logits = tout / n_heads[-1]
        return logits

    # def loss():

