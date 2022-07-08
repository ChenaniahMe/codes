import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description='PerNetGNN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='PubMed')
    parser.add_argument('--model', type=str, default='PerNet')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--tsne', type=bool, default=True)
    parser.add_argument('--extend_dim', type=int, default=16, help="the extended dimension needs to be an integer multiple of blocks")
    parser.add_argument('--enhance_range', type=int, default=5, help="the number of generated parameter-enhanced pools")
    parser.add_argument('--hidden_channels', type=int, default=100, help="hidden_channels must be an integer multiple of block_num")
    parser.add_argument('--block_num', type=int, default=4, help="the number of splitting weights,hidden_channels must be an integer multiple of block_num")
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default: fix split')
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--K', type=int, default=20)
    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    
    return args

