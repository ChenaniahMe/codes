import imp
import torch
import argparse
import config
import numpy as np
from parse import parse_args
args = parse_args()

def masked_softmax_cross_entropy(logits, labels):
    y = torch.softmax(logits,dim=1)
    tf_log = torch.log(y)
    pixel_wise_mult = labels*tf_log
    loss = -pixel_wise_mult
    loss = torch.sum(loss,axis=1)
    return torch.mean(loss)

def featureswq(features, weights,queues):
    '''PerMM Modules
    '''
    def wpools(tfeatures):
        pools = []
        # for i in range(len(tfeatures)):
        #     for j in range(len(tfeatures)):
        for i in range(args.enhance_range):
            for j in range(i+1,args.enhance_range):
                if i != j:
                    pools.append(tfeatures[i]*tfeatures[j])
        return pools

    if queues==1:
        features = features@weights

        #split features
        nfeatures = []
        for tb in torch.split(features,1,dim=1):
            nfeatures.append(tb)

        #ehance features
        pfeatures = wpools(nfeatures)
        index = np.random.randint(0,len(pfeatures), args.extend_dim//args.block_num)
        for i in range(len(index)):
            nfeatures.append(pfeatures[index[i]])
            
    elif queues==2:
        # print("features.shape",features.shape)
        # print("weights.shape",weights.shape)
        features = features@weights
        nfeatures = features
        return nfeatures
    return torch.cat(nfeatures,axis=-1)



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'{self.info} Run {run + 1:02d}:')
            # print("result",result)
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            # import ipdb; ipdb.set_trace()
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'{self.info} All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


    def best_result(self, run=None, with_var=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            train1 = result[:, 0].max()
            valid  = result[:, 1].max()
            train2 = result[argmax, 0]
            test   = result[argmax, 2]
            return (train1, valid, train2, test)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[argmax, 0].item()
                test = r[argmax, 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            train1 = r.mean().item()
            train1_var = f'{r.mean():.2f} ± {r.std():.2f}'
            
            r = best_result[:, 1]
            valid = r.mean().item()
            valid_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 2]
            train2 = r.mean().item()
            train2_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 3]
            test = r.mean().item()
            test_var = f'{r.mean():.2f} ± {r.std():.2f}'

            if with_var:
                return (train1, valid, train2, test, train1_var, valid_var, train2_var, test_var)
            else:
                return (train1, valid, train2, test)