import torch
import torch_sparse
from torch_sparse import SparseTensor, matmul

size = 4
row_index = [0,0,1,2,1,3,3]
col_index = [1,2,0,0,3,1,3]
row_index = torch.Tensor(row_index).long()
col_index = torch.Tensor(col_index).long()
mask = row_index >= col_index
row_index = row_index[mask]
col_index = col_index[mask]
edge_num = row_index.numel()
row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)])
col = torch.cat([row_index, col_index])
value = torch.cat([torch.ones(edge_num), -1*torch.ones(edge_num)])
inc = SparseTensor(row=row, rowptr=None, col=col, value=value,sparse_sizes=(edge_num, size))
print("inc.to_dense()",inc.to_dense())