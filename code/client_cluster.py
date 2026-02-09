import os
import torch
from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import socket
import time
from datetime import datetime
import threading
from transformers import TrainingArguments, Trainer

import sys
sys.path.append('utils')
from prepare_model_params import convert_model_key_to_idx
from build_model import build_model
from communication import send, recv
from general_utils import set_seed
from evaluation import calculate_SLC_metrics, display_results

EPS = 1e-7


def request(src_addr, tgt_addr, send_msg, buffer_size, timeout):
    while True:
        try:
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            soc.bind(src_addr)
            soc.settimeout(None)
            soc.connect(tgt_addr)
            print(f"Run a Thread for connection with {tgt_addr}. Send {round(len(send_msg) * 1e-9, 4)} Gb.")
            break
        except ConnectionRefusedError as ex:
            print(f'Client {src_addr}', ex)
            break
        except Exception as ex:
            print(ex)

    try:
        send(soc, send_msg, buffer_size)

        time_struct = time.gmtime()
        date_time = f"Waiting for data from {tgt_addr}. Starting at {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec}"
        print(date_time)
        msg, status = recv(soc, buffer_size, timeout)
        print(f"Receive {msg['subject'].upper()} message from {tgt_addr}")
        if status == 0:
            print(
                f"Connection Closed with {tgt_addr} either due to inactivity for {timeout} sec or an error.")

        return msg

    except BaseException as e:
        print(f"Error Connecting to Server {tgt_addr}: {e}")

    finally:
        soc.close()
        print(f'Close connection with {tgt_addr}.')


class ClientCluster():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(None)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.server_addr = args.server_addr
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.port = args.port
        print('address:', (self.ip, self.port))

    def register_task(self, args, server_args, global_keys, recv_model_params):
        args.task = server_args.task
        args.n_class = server_args.n_class
        args.epochs = server_args.epochs
        args.model_name = server_args.model_name

        peft_config = None
        set_seed(server_args.seed) # maintain the same data distirbution across methods and with sync FL
        if server_args.task.startswith('cifar'):
            from load_cifar import load_noniid_cifar
            trainData, valData, testData = load_noniid_cifar(server_args.task, os.path.join(args.data_dir, server_args.task), server_args.data_shares, server_args.alpha)
            self.collate_fn = None

        elif server_args.task == 'mnist':
            from load_mnist import load_noniid_mnist
            trainData, testData = load_noniid_mnist(os.path.join(args.data_dir, server_args.task), server_args.data_shares, server_args.alpha)
            self.collate_fn = None

        elif server_args.task == 'har':
            from load_har import load_data, collate_fn
            trainData, testData = load_data(os.path.join(args.data_dir, args.task))
            self.collate_fn = collate_fn

        elif server_args.task == '20_newsgroups':
            from load_20_newsgroups import load_noniid_data
            trainData, testData, self.collate_fn = load_noniid_data(server_args.base_model_name, server_args.data_shares, server_args.alpha)
            args.lr = 5e-5
            peft_config = server_args.peft_config

        else:
            raise ValueError('Wrong dataset.')

        self.trainData = trainData
        self.testData = testData
        self.n_epochs = 0

        self.model = build_model(args.model_name, args.task, args.n_class, peft_config, server_args.base_model_name).to(torch.device('cpu'))

        init_model_params = {global_keys[k_idx]: p.cpu() for k_idx, p in recv_model_params.items()}
        missing_keys, unexpected_keys = self.model.load_state_dict(init_model_params, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))


    def run(self, args):
        print('---Start Registration---')
        # receive message from server
        send_msg = pickle.dumps({"subject": "register", "data": {"client_port": [args.port + 1 + i for i in range(args.n_clients)]}})
        recv_data = request((self.ip, self.port), self.server_addr, send_msg, self.buffer_size, self.timeout)
        self.register_task(args, recv_data['data']['server_args'], recv_data['data']['global_keys'], recv_data['data']['model_params'])

        threads = []
        print("received client ids:", recv_data['data']['cid'])
        for cid in recv_data['data']['cid']:
            t = threading.Thread(target=self.client_thread, args=(args, cid, recv_data['data']['global_keys'], recv_data['data']['trainable_params']))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        time.sleep(30)

    def client_thread(self, args, cid, global_keys, trainable_params):
        client_args = deepcopy(args)
        client_args.port = args.port + 1 + cid
        client_args.device = torch.device(f"cuda:{cid % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        client = Client(client_args, cid, self.trainData[cid], self.testData, self.collate_fn, self.model, global_keys, trainable_params)
        client.train(client_args)


class Client():
    def __init__(self, args, cid, trainData, testData, collate_fn, model, global_keys, trainable_params):
        self.cid = cid
        time.sleep(60 * self.cid / 5.)  # avoid the congestion of GPUs
        print(f'#### Create Client {cid} ####')
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(None)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.server_addr = args.server_addr
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.port = args.port
        print('address:', (self.ip, self.port))

        self.model = deepcopy(model)
        self.global_keys = global_keys
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}
        self.trainable_params = trainable_params

        self.trainData = trainData
        self.testData = testData
        self.collate_fn = collate_fn
        self.device = args.device

        class_distribution = np.zeros(args.n_class)
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        for batch in train_loader:
            if args.model_name.find('BERT') != -1:
                labels = batch["labels"].numpy().tolist()
            else:
                labels = batch[1].numpy().tolist()

            for cls in range(args.n_class):
                class_distribution[cls] += labels.count(cls)

        self.class_distribution = class_distribution / np.sum(class_distribution)
        print(f'Client {self.cid} n_train: {len(self.trainData)}, n_class: {args.n_class}')
        print(f'Client {self.cid} class distribution:', list(self.class_distribution))

        if args.model_name.find('BERT') != -1:
            self.training_args = TrainingArguments(
                output_dir=f'./client{self.cid}_results',
                overwrite_output_dir=True,
                learning_rate=args.lr,  # 5e-5
                num_train_epochs=args.epochs,
                per_device_train_batch_size=8
            )

    def train_one_batch(self, model, sample, label, optimizer, global_params):
        model.train()

        criterion = nn.CrossEntropyLoss()
        label = label.to(self.device, dtype=torch.long)
        if len(label.shape) > 1:
            label = torch.argmax(label, dim=-1)

        optimizer.zero_grad()
        intermediate_out, out = model(sample.to(device=self.device), return_emb=True)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()

        return loss.item()

    def project_to_orthogonal(self, orig_grad, other_grads):
        # other_grads are orthogonal to each other
        proj_grad = deepcopy(orig_grad)
        # remove all projections on the directions that needs to be orthogonal
        for o_vec in other_grads:
            o_vec = o_vec.to(self.device)
            dot_product = torch.dot(orig_grad.flatten(), o_vec.flatten())
            norm_o_squared = torch.dot(o_vec.flatten(), o_vec.flatten()) + 1e-19
            proj_grad -= (o_vec * dot_product / norm_o_squared).view(*orig_grad.size())

        return proj_grad

    def local_update(self, args):
        recv_params = {k: p.cpu() for k, p in self.model.state_dict().items()}

        self.model = self.model.to(self.device)
        if args.model_name.find("BERT") != -1:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.trainData,
                data_collator=self.collate_fn,
            )
            self.trainer.train()
        else:
            train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=1)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            for e in range(args.epochs):
                self.n_epochs += 1
                start_time = datetime.now()
                for sample, label in train_loader:
                    self.train_one_batch(self.model, sample, label, optimizer, recv_params)

                end_time = datetime.now()
                duration = (end_time - start_time).seconds / 60.
                print('[TRAIN] Client %i, Epoch %i, time=%.3fmins' % (self.cid, self.n_epochs, duration))

        curr_params = {k: p.cpu() for k, p in self.model.state_dict().items() if k in self.trainable_params}

        delta_weights = {}
        for k in self.trainable_params:
            if k.find('num_batches_tracked') != -1 or k.find('running_mean') != -1 or k.find('running_var') != -1:
                delta_weights[k] = curr_params[k].cpu()
            else:
                delta_weights[k] = recv_params[k].cpu() - curr_params[k].cpu()

        self.model = self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        return delta_weights, curr_params


    def train(self, args):
        # types of messenge that server send to client
        # train: ask client to train model and return the model parameter
        # update: send the updated model to the client
        # stop: ask client to stop training and close connection
        self.n_epochs = 0
        print('--- Start Training ---')

        while True:
            # local_update
            delta_weights, updated_weights = self.local_update(args)
            test_scores, _ = self.evaluate(args, self.model)
            print(f'-----Test Client {self.cid} model-----')
            display_results(test_scores, ['ACC'])

            send_msg = pickle.dumps({"subject": "update", "data": {
                "model_params": convert_model_key_to_idx(self.global_key_to_idx, delta_weights),
                "full_params": convert_model_key_to_idx(self.global_key_to_idx, updated_weights),
                "test_scores": test_scores
            }})

            # update global model weight
            recv_data = request((self.ip, self.port), self.server_addr, send_msg, self.buffer_size, self.timeout)
            updated_model_params = {self.global_keys[k_idx]: p.cpu() for k_idx, p in recv_data['data']['model_params'].items()}

            missing_keys, unexpected_keys = self.model.load_state_dict(updated_model_params, strict=False)
            if len(missing_keys) or len(unexpected_keys):
                print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))

            if recv_data['subject'] == 'stop':
                break

    def evaluate(self, args, model):
        if args.task == '20_newsgroups':
            batch_size = 8
        else:
            batch_size = args.batch_size
        data_loader = DataLoader(self.testData, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
        criterion = nn.CrossEntropyLoss()
        y_pred = []
        y_true = []

        model = model.to(self.device)
        model.eval()
        avg_loss = []
        with torch.no_grad():
            for batch in data_loader:
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

