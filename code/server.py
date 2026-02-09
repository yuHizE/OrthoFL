import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle
import socket
import threading
import time
import sys
from copy import deepcopy
from peft import LoraConfig
from tqdm import tqdm

sys.path.append('utils')
from prepare_model_params import convert_model_key_to_idx
from general_utils import set_seed
from delay_simulation import get_delay, get_update_sequence
from communication import recv, send
from build_model import build_model
from model_aggregation import FedAsync, OrthoFL
from evaluation import calculate_SLC_metrics, display_results

EPS = 1e-7

class Server():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(None)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = args.port
        self.total_clients = args.total_clients
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.device = args.device
        self.metrics = args.metrics
        self.clients = {} # address to cid
        self.client_addr = []
        self.logger = args.logger

        args.peft_config = None
        set_seed(args.seed)
        if args.task.startswith('cifar'):
            from load_cifar import load_noniid_cifar
            _, _, self.testData = load_noniid_cifar(args.task, os.path.join(args.data_dir, args.task), args.data_shares, args.alpha)
            self.collate_fn = None

        elif args.task == 'mnist':
            from load_mnist import load_noniid_mnist
            _, self.testData = load_noniid_mnist(os.path.join(args.data_dir, args.task), args.data_shares, args.alpha)
            self.collate_fn = None

        elif args.task == '20_newsgroups':
            from load_20_newsgroups import load_noniid_data
            _, self.testData, self.collate_fn = load_noniid_data("distilbert-base-uncased", args.data_shares, args.alpha)
            args.peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_lin", "v_lin"]
            )

        elif args.task == 'har':
            from load_har import load_data, collate_fn
            _, self.testData = load_data(os.path.join(args.data_dir, args.task))
            self.collate_fn = collate_fn
        else:
            raise ValueError('Wrong dataset.')

        # the first in the list is the final evaluated one
        self.global_model = build_model(args.model_name, args.task, args.n_class, args.peft_config, args.base_model_name).to(self.device)
        print(self.global_model)
        if args.start_round != 0:
            state_dict = torch.load(os.path.join(args.model_dir, f'global_model_alpha{args.alpha}_tc{args.total_clients}_{(args.start_round-1)}.pth'))
            self.global_model.load_state_dict(state_dict)
        self.logger.critical(f'Global model loaded from f{args.model_dir}global_model_alpha{args.alpha}_tc{args.total_clients}_{(args.start_round-1)}.pth')
        self.logger.debug(f'Global model created')

        # create global_model_params after activate adapters
        self.global_model_params = {k: p.cpu() for k, p in self.global_model.state_dict().items()}

        # load checkpoint
        if args.start_round != 0:
            self.aggregated_model_params = torch.load(os.path.join(args.model_dir, f'avg_model_alpha{args.alpha}_tc{args.total_clients}_{(args.start_round-1)//1000}.pth'))
        else:
            self.aggregated_model_params = deepcopy(self.global_model_params)

        self.global_keys = list(self.global_model_params.keys())
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}

        self.trainable_params = self.global_keys
        self.aggregator = OrthoFL(self.global_keys)
        self.aggregator_client = FedAsync(self.global_keys, args.total_clients, alpha=0.6, strategy="polynomial", a=0.5)
        if args.start_round != 0:
            self.aggregator = pickle.load(open(os.path.join(args.model_dir, f'aggregator_alpha{args.alpha}_tc{args.total_clients}_{(args.start_round-1)//1000}.pkl'), 'rb'))

        if args.model_name.find('lora') != -1:
            self.trainable_params = []
            for k, p in self.global_model.named_parameters():
                if p.requires_grad:
                    self.trainable_params.append(k)

            print('# trainable params:', len(self.trainable_params), self.global_model.print_trainable_parameters())

    def register_client(self, address):
        self.client_addr.append(address)

    def run(self, args):
        self.device = args.device
        self.start_training_model_params = {cid: deepcopy(self.global_model_params) for cid in range(args.total_clients)}

        np.random.seed(args.seed)
        delay_mean, delay_std = get_delay(n_clients=args.total_clients)
        self.update_sequence, self.update_timestamp = get_update_sequence(n_updates=args.rounds, delay_mean=delay_mean, delay_std=delay_std)
        self.logger.critical(f'Update sequence: {self.update_sequence[:100]}...')
        # waiting for server to send request
        try:
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            soc.bind((self.ip, self.port))
            soc.listen(self.total_clients)
            self.logger.debug('Start Listening...')

            self.n_updates = 0
            if args.start_round != 0:
                self.n_updates = args.start_round
                self.update_sequence = self.update_sequence[args.start_round:]
                self.update_timestamp = self.update_timestamp[args.start_round:]
            threads = []
            while True:
                if not len(self.update_sequence):
                    break

                new_socket, source_addr = soc.accept()
                new_socket.settimeout(args.timeout)

                # receive request
                msg, status = recv(new_socket, args.buffer_size, recv_timeout=60)

                if isinstance(msg, dict):
                    # todo: tell the client if the update batch normalization layer
                    if msg['subject'] == 'register':
                        send_msg = {
                            "subject": "register",
                            "data": {
                                "server_args": args,
                                "global_keys": self.global_keys,
                                "model_params": convert_model_key_to_idx(self.global_key_to_idx, {k: v for k, v in self.global_model_params.items() if k in self.trainable_params}),
                                "trainable_params": self.trainable_params
                            }}
                        if isinstance(msg['data']['client_port'], list): # a list of clients. sent from client cluster
                            for port in msg['data']['client_port']:
                                self.client_addr.append((source_addr[0], port))
                            for cid, addr in enumerate(self.client_addr):
                                self.clients[addr] = cid
                            send_msg['data']['cid'] = [self.clients[(source_addr[0], port)] for port in msg['data']['client_port']]

                        elif isinstance(msg['data']['client_port'], int):
                            self.client_addr.append((source_addr[0], msg['data']['client_port']))
                            for cid, addr in enumerate(self.client_addr):
                                self.clients[addr] = cid
                            send_msg['data']['cid'] = self.clients[source_addr]

                        data_byte = pickle.dumps(send_msg)
                        send(new_socket, data_byte, args.buffer_size)

                        del data_byte

                    elif msg['subject'] == 'update':
                        agg_thread = threading.Thread(target=self.aggregate, args=(args, new_socket, source_addr, msg))
                        agg_thread.start()
                        threads.append(agg_thread)

            for agg_thread in threads:
                agg_thread.join()

        finally:
            soc.close()


    def evaluate(self, args, model):
        data_loader = DataLoader(self.testData, batch_size=args.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
        criterion = nn.CrossEntropyLoss()
        y_pred = []
        y_true = []

        model = model.to(self.device)
        model.eval()
        avg_loss = []
        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader)):
                if args.model_name.find('BERT') != -1:
                    out = model(**batch.to(self.device)).logits
                    label = batch["labels"].to(self.device, dtype=torch.float)
                else:
                    sample, label = batch[0], batch[1]
                    out = model(sample.to(self.device), return_emb=False)
                    label = label.to(self.device, dtype=torch.float)

                if len(label.shape) == 1:
                    label = F.one_hot(label.to(torch.long), num_classes=args.n_class)

                avg_loss.append(criterion(out, torch.argmax(label, dim=-1)).item())
                out = torch.softmax(out, dim=-1)

                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_scores = calculate_SLC_metrics(y_true, y_pred)

        return test_scores, np.mean(avg_loss)


    def aggregate(self, args, socket, source_addr, msg):
        while self.n_updates < args.rounds:
            if self.clients[source_addr] == self.update_sequence[0]:
                full_model_params = {self.global_keys[k_idx]: p for k_idx, p in msg['data']['full_params'].items()}
                client_scores = msg['data']['test_scores']['ACC'] if 'test_scores' in msg['data'] else np.nan

                updated_model_params = self.aggregator.update(self.clients[source_addr], self.start_training_model_params[self.clients[source_addr]], self.global_model_params, full_model_params)
                self.aggregated_model_params = self.aggregator_client.update(self.clients[source_addr], self.aggregated_model_params, full_model_params)

                # evaluate global model (moving average)
                avg_model = deepcopy(self.global_model)
                avg_model.load_state_dict(self.aggregated_model_params, strict=False)
                test_scores, _ = self.evaluate(args, avg_model)

                # load params into global model. this model is sent to client for subsequent training
                self.global_model.load_state_dict(updated_model_params, strict=False)
                self.global_model_params = {k: p.cpu() for k, p in self.global_model.state_dict().items()}

                # display_results(test_scores, self.metrics)
                self.logger.critical('[TRAIN] Updates %i, Timestamp %.3f(s), Client %i, Global ACC=%.4f' % (
                        self.n_updates, self.update_timestamp.pop(0), self.clients[source_addr], test_scores['ACC']))

                # reply request
                if self.n_updates < args.rounds:
                    subject = 'update'
                else:
                    subject = 'stop'
                send_msg = {
                    "subject": subject,
                    "data": {"model_params": convert_model_key_to_idx(self.global_key_to_idx, {k: v for k, v in self.global_model_params.items() if k in self.trainable_params})}}

                data_byte = pickle.dumps(send_msg)
                self.start_training_model_params[self.clients[source_addr]] = deepcopy(self.global_model_params)
                socket.settimeout(None)
                send(socket, data_byte, args.buffer_size)

                socket.close()
                torch.save(self.global_model_params, os.path.join(args.model_dir, f'global_model_alpha{args.alpha}_tc{args.total_clients}_{self.n_updates // 1000}.pth'))
                torch.save(self.aggregated_model_params, os.path.join(args.model_dir, f'avg_model_alpha{args.alpha}_tc{args.total_clients}_{self.n_updates // 1000}.pth'))
                pickle.dump(self.aggregator, open(os.path.join(args.model_dir, f'aggregator_alpha{args.alpha}_tc{args.total_clients}_{self.n_updates // 1000}.pkl'), 'wb'))

                del data_byte
                self.n_updates += 1
                self.update_sequence.pop(0)
                break

            else:
                time.sleep(5)
