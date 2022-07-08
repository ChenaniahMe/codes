import torch
import torch.nn.functional as F
import random

import time

from util import Logger
from pernet import get_model
from train_eval import train, test
import numpy as np
from parse import parse_args

def main():
    args = parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)

    total_start = time.perf_counter()

    if 'adv' in args.dataset:
        from dataset_adv import get_dataset
    else:
        from dataset import get_dataset

    ## data split
    for split in range(random_split_num):
        dataset, data, split_idx = get_dataset(args, split)
        train_idx = split_idx['train']
        data = data.to(device)
        print("device",device)
        print("Data:", data)
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()

        model = get_model(args, dataset, data)
        print(model)
        for param in model.named_parameters():
            print(param[0], param[1].requires_grad)
        print("param is over")
        ## multiple run for each split
        for run in range(args.runs):
            runs_overall = split * args.runs + run
            # model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            t_start = time.perf_counter()
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                # print("Train"*20)
                loss = train(model, data, train_idx, optimizer)
                # print("Test"*20)
                result = test(model, data, split_idx, epoch)
                logger.add_result(runs_overall, result)
                
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')

            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)

    total_end = time.perf_counter()
    total_duration = total_end - total_start
    print('total time: ', total_duration)
    logger.print_statistics()

if __name__ == "__main__":
    main()

