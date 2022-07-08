import imp
import torch
import torch.nn.functional as F
import numpy as np
from ogb.nodeproppred import Evaluator
from parse import parse_args
from util import masked_softmax_cross_entropy

args = parse_args()

def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out,tsen= model(data=data)
    out = out[train_idx]
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    
    # print("data.train_mask",data.train_mask)
    # print("train_idx.shape",train_idx)
    # print("out",out)
    # print("y",y)
    loss = F.nll_loss(out, y)
    # loss = masked_softmax_cross_entropy(out, F.one_hot(y))

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx,epoch):
    model.eval()
    out,tsen= model(data=data)
    
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    if epoch==args.epochs and args.tsne:
        save_ytrue = y[split_idx['test']]
        save_ypred = y_pred[split_idx['test']]
        save_tsen = tsen[split_idx['test']]

        np.save("../tsen/"+args.dataset+"_ytrue.npy",save_ytrue.cpu())
        np.save("../tsen/"+args.dataset+"_ypred.npy",save_ypred.cpu())
        np.save("../tsen/"+args.dataset+"_tsen.npy",save_tsen.cpu())
        print("have saved")
    # print("save_ytrue.shape",save_ytrue.shape)
    # print("save_ypred.shape",save_ypred.shape)
    # print("save_tense.shape",save_tsen.shape)
    return train_acc, valid_acc, test_acc