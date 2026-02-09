import os
import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import argparse
import time

from server import Server
sys.path.append('utils')
from logger import Logger
from general_utils import set_seed

def main(args):
    log_dir = os.path.join(args.save_dir, f"logs/{args.task}/seed{args.seed}/")
    args.model_dir = os.path.join(args.save_dir, f"saved_weights/{args.task}/seed{args.seed}/")
    args.data_shares = [1 / (args.total_clients)] * (args.total_clients)
    assert args.total_clients == len(args.data_shares)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    args.log_path = os.path.join(log_dir, f"alpha{args.alpha}.tc{args.total_clients}.log")  # + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    args.logger = Logger(file_path=args.log_path).get_logger()
    args.logger.critical(args.log_path)

    if args.device is None:
        args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    if args.device == torch.device("cuda") and args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    args.base_model_name = "bert-base-uncased"
    set_seed(args.seed)
    if args.task == 'cifar10':
        args.model_name = 'VGG11'
    elif args.task == 'cifar100':
        args.model_name = 'MobileNetV2'
    elif args.task == 'mnist':
        args.model_name = 'LeNet5'
    elif args.task == '20_newsgroups':
        args.model_name = 'DistilBERT_lora'
        args.base_model_name = 'distilbert-base-uncased'
    elif args.task == 'har':
        args.model_name = 'ResNet1D'

    args.metrics = ['F1', 'AUC', 'ACC']

    if args.task == 'har':
        args.n_class = 6
    elif args.task == 'cifar10':
        args.n_class = 10
    elif args.task == 'cifar100':
        args.n_class = 100
    elif args.task == 'mnist':
        args.n_class = 10
    elif args.task == '20_newsgroups':
        args.n_class = 20
        args.rank = 8
        args.lora_alpha = 16
    args.logger.critical(args)

    server = Server(args)
    args.logger.debug('Server created.')

    server.run(args)

    del args
    del server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4321, help="random seed")
    parser.add_argument('-t', '--task', choices=['har', 'cifar10', 'cifar100', 'mnist', '20_newsgroups'], default='cifar10', help="task name")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="gpu id")
    parser.add_argument('--port', type=int, default=13456)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--data_dir', type=str, default="data/")
    # training & communication
    parser.add_argument('--wait_time', type=int, default=180, help="time (seconds) to wait for start training (all clients start training at the same time)")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="use cuda or cpu")
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha for dirichlet distribution")
    parser.add_argument('--total_clients', type=int, default=None, help="number of total clients")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of training epochs per round")
    parser.add_argument('-r', '--rounds', type=int, default=50, help="number of communication rounds")
    parser.add_argument('--buffer_size', type=int, default=1048576)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--start_round', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.data_dir):
        if not os.path.exists(args.data_dir):
            raise FileExistsError(f'Not Found {args.data_dir}')

    if args.task == 'cifar10':
        args.rounds = 500
    elif args.task == 'cifar100':
        args.rounds = 100000
    elif args.task == 'mnist':
        args.rounds = 500
    elif args.task == '20_newsgroups':
        args.rounds = 200
    elif args.task == 'har':
        args.rounds = 1000

    if args.total_clients is None: # default
        if args.task == 'har':
            args.total_clients = 21
        elif args.task == 'cifar10':
            args.total_clients = 10
        elif args.task == 'cifar100':
            args.total_clients = 100
        elif args.task == 'mnist':
            args.total_clients = 10
        elif args.task == '20_newsgroups':
            args.total_clients = 20

    main(args)
    time.sleep(10)
