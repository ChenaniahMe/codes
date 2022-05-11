import logging
import argparse
import os
import numpy as np
import torch
import pandas as pd

from ppnp.pytorch import PPNP
from ppnp.pytorch.training import train_model
from ppnp.pytorch.earlystopping import stopping_args
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset

logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)

# 1
# 2
graph_name = 'cora_ml'  # ['cora_ml', 'citeseer', 'pubmed']
graph = load_dataset(graph_name)
graph.standardize(select_lcc=True)

# 3
test = True

test_seeds = [
        2144199730,  794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407,  635625077, 3538425002,  960893189,
        497096336, 3940842554, 3594628340,  948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]
val_seeds = [
        2413340114, 119332950, 1789234713, 2222151463, 2813247115,
        1920426428, 4272044734, 2092442742, 841404887, 2188879532,
        646784207, 1633698412, 2256863076,  374355442,  289680769,
        4281139389, 4263036964,  900418539,  3258769933, 1628837138]

if test:
    seeds = test_seeds
else:
    seeds = val_seeds

if graph_name == 'microsoft_academic':
    nknown = 5000
else:
    nknown = 1500

nknown = len(graph.labels) - 1000
idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': nknown}

# 4
if graph_name == 'microsoft_academic':
    alpha = 0.2
else:
    alpha = 0.1

prop_ppnp = PPRExact(graph.adj_matrix, alpha=alpha)
prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=alpha, niter=10)

# 5
model_args = {
    'hiddenunits': [64],
    'drop_prob': 0.5,
    'propagation': prop_appnp}

reg_lambda = 5e-3
learning_rate = 0.01

# 6
niter_per_seed = 3
seeds = seeds[:10]
save_result = False
print_interval = 100
device = 'cuda:3'

results = []
niter_tot = niter_per_seed * len(seeds)
i_tot = 0
with pd.ExcelWriter(f'./Appnp_1000.xlsx') as writer:
    for graph_name in ['cora', 'citeseer', 'pubmed']:  #['cora', 'citeseer', 'pubmed']
        graph = load_dataset(graph_name)
        graph.standardize(select_lcc=True)
        prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=alpha, niter=10)
        model_args['propagation'] = prop_appnp
        for column, seed in enumerate(seeds):
            idx_split_args['seed'] = seed

            i_tot += 1
            logging_string = f"Iteration {i_tot} of {niter_tot}"
            logging.log(22,
                    logging_string + "\n                     "
                    + '-' * len(logging_string))
            _, result = train_model(
                graph_name, writer, column, PPNP, graph, model_args, learning_rate, reg_lambda,
                idx_split_args, stopping_args, test, device, None, print_interval)
            results.append({})
            results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
            results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
            results[-1]['runtime'] = result['runtime']
            results[-1]['runtime_perepoch'] = result['runtime_perepoch']
            results[-1]['split_seed'] = seed

# 7
# import pandas as pd
# import seaborn as sns
#
# result_df = pd.DataFrame(results)
# result_df.head()
#
# def calc_uncertainty(values: np.ndarray, n_boot: int = 1000, ci: int = 95) -> dict:
#     stats = {}
#     stats['mean'] = values.mean()
#     boots_series = sns.algorithms.bootstrap(values, func=np.mean, n_boot=n_boot)
#     stats['CI'] = sns.utils.ci(boots_series, ci)
#     stats['uncertainty'] = np.max(np.abs(stats['CI'] - stats['mean']))
#     return stats
#
# stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
# valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
# runtime = calc_uncertainty(result_df['runtime'])
# runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])
#
# print("APPNP\n"
#       "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n"
#       "{}: Accuracy: {:.2f} ± {:.2f}%\n"
#       "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms"
#       .format(
#           stopping_acc['mean'] * 100,
#           stopping_acc['uncertainty'] * 100,
#           'Test' if test else 'Validation',
#           valtest_acc['mean'] * 100,
#           valtest_acc['uncertainty'] * 100,
#           runtime['mean'],
#           runtime['uncertainty'],
#           runtime_perepoch['mean'] * 1e3,
#           runtime_perepoch['uncertainty'] * 1e3,
#       ))