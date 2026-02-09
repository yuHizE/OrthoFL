import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import time

from client_cluster import ClientCluster


def parse_args():
    # default setting is for extrasensory
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clients', type=int, default=10, help="number of clients")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('-s', '--server_addr', type=str, help="server address", required=True)
    parser.add_argument('-g', '--gpus', type=list, default=[4, 7], help="gpu id")
    parser.add_argument('--port', type=int, default=8361)
    parser.add_argument('--server_port', type=int, default=12345)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--buffer_size', type=int, default=1048576, help="initial buffer size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate of local model")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.data_dir):
        if not os.path.exists(args.data_dir):
            raise FileExistsError(f'Not Found {args.data_dir}')

    args.server_addr = (args.server_addr, args.server_port)
    print(args)
    print(f'#### Run Client ####')
    client_cluster = ClientCluster(args)
    client_cluster.run(args)
    time.sleep(30)