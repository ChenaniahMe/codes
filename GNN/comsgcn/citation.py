import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import pandas as pd

# Arguments
args = get_citation_args()
experi_num = 10

with pd.ExcelWriter(f'./SGC_mgl.xlsx') as writer:
    for args.dataset in ['cora']: # ['cora', 'citeseer', 'pubmed']
        for iii in range(experi_num):
            if args.tuned:
                if args.model == "SGC":
                    with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
                        args.weight_decay = pkl.load(f)['weight_decay']
                        print("using tuned weight decay: {}".format(args.weight_decay))
                else:
                    raise NotImplemented

            # setting random seeds
            # set_seed(args.seed, args.cuda)

            adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

            # 此处表示是否使用新分布的数据形式
            # idxes = torch.randperm(features.size(0))
            # idx_train, idx_val, idx_test = idxes[:-1000], idxes[-1000:-500], idxes[-500:]


            model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

            if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
            print("{:.4f}s".format(precompute_time))

            def train_regression(model,
                                 train_features, train_labels,
                                 val_features, val_labels,
                                 test_features, test_labels,
                                 epochs=args.epochs, weight_decay=args.weight_decay,
                                 lr=args.lr, dropout=args.dropout):

                optimizer = optim.Adam(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)
                t = perf_counter()
                res = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    output = model(train_features)
                    loss_train = F.cross_entropy(output, train_labels)
                    train_acc = accuracy(output, train_labels)
                    res['train_loss'].append(float(loss_train.cpu()))
                    res['train_acc'].append(train_acc.cpu())
                    loss_train.backward()
                    optimizer.step()
                    with torch.no_grad():
                        model.eval()
                        output = model(val_features)
                        val_loss = F.cross_entropy(output, val_labels)
                        val_acc = accuracy(output, val_labels)
                        output = model(test_features)
                        test_loss = F.cross_entropy(output, test_labels)
                        test_acc = accuracy(output, test_labels)
                        res['val_loss'].append(val_loss.cpu())
                        res['val_acc'].append(val_acc.cpu())
                        res['test_loss'].append(test_loss.cpu())
                        res['test_acc'].append(test_acc.cpu())
                train_time = perf_counter()-t

                res = {key: np.array(res[key]) for key in res.keys()}
                for sheet in res:
                    temp = pd.Series(res[sheet])
                    temp.to_excel(writer, sheet_name=args.dataset+'_'+sheet,
                                  header=False, index=False, startcol=iii)
                # writer.save()
                # writer.close()

                with torch.no_grad():
                    model.eval()
                    output = model(val_features)
                    acc_val = accuracy(output, val_labels)

                return model, acc_val, train_time

            def test_regression(model, test_features, test_labels):
                model.eval()
                return accuracy(model(test_features), test_labels)

            if args.model == "SGC":
                model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                 features[idx_test], labels[idx_test], args.epochs, args.weight_decay, args.lr, args.dropout)
                acc_test = test_regression(model, features[idx_test], labels[idx_test])

            print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
            print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
