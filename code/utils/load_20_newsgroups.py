from datetime import datetime
import random
import numpy as np
from torch.utils.data import Subset
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding



def load_noniid_data(base_model, data_shares, alpha):
    # alpha: parameter for dirichlet distribution
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    datasets = load_dataset('SetFit/20_newsgroups')
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text", "label_text"])
    testData = tokenized_datasets['test'] #.shard(num_shards=20, index=0) # todo
    train_data = tokenized_datasets['train'] #.shard(num_shards=20, index=0) # todo

    num_classes, num_samples, data_labels_list = get_num_classes_samples(train_data)
    q_class = np.random.dirichlet([alpha] * num_classes, len(data_shares))
    data_class_idx = {j: np.where(data_labels_list == j)[0] for j in range(num_classes)}
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    q_client = np.array(data_shares, dtype=float) / np.sum(data_shares)
    usr_subset_idx = gen_data_split(data_class_idx, q_class, q_client)

    trainData = list(map(lambda x: Subset(train_data, x), usr_subset_idx))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    return trainData, testData, data_collator


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    data_labels_list = []
    for d in dataset:
        data_labels_list.append(d['label'])
    data_labels_list = np.array(data_labels_list)
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)

    return num_classes, num_samples, data_labels_list


def gen_data_split(data_class_idx, q_class, q_client, k_samples_at_a_time=3, timeout=180):
    """Non-iid Dirichlet partition.
    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients.
    Args:
        :param q_class: class distribution at each client
        :param q_client: sample distribution cross clients
    Returns:
        dict: ``{ client_id: indices}``.
    """
    num_samples = np.array([len(indices) for cls, indices in data_class_idx.items()])
    num_samples_clients = (q_client * num_samples.sum()).round().astype(int)
    delta_data = num_samples.sum() - num_samples_clients.sum()
    client_id = 0
    for i in range(abs(delta_data)):
        num_samples_clients[client_id % len(q_client)] += np.sign(delta_data)
        client_id += 1

    # Create class index mapping
    data_class_idx = {cls: set(data_class_idx[cls]) for cls in data_class_idx}

    q_class_cumsum = np.cumsum(q_class, axis=1) # cumulative sum
    num_samples_tilde = deepcopy(num_samples)

    client_indices = [[] for _ in range(len(q_client))]

    start_time = datetime.now()
    while np.sum(num_samples_clients) != 0:
        if (datetime.now() - start_time).seconds > timeout:
            print(f'Timeout. {num_samples_clients.sum()} samples remain.')
            break
        # iterate clients
        curr_cid = np.random.randint(len(q_client))
        # If current node is full resample a client
        if num_samples_clients[curr_cid] <= 0:
            continue

        while True:
            if (datetime.now() - start_time).seconds > timeout:
                print(f'Timeout. {num_samples_clients.sum()} samples remain.')
                break

            curr_class = np.argmax((np.random.uniform() <= q_class_cumsum[curr_cid]) & (num_samples_tilde > 0))
            # Redraw class label if no rest in current class samples
            if num_samples_tilde[curr_class] <= 0:
                continue

            k_samples = min(k_samples_at_a_time, len(data_class_idx[curr_class]))
            random_sample_idx = np.random.choice(list(data_class_idx[curr_class]), k_samples, replace=False)
            num_samples_tilde[curr_class] -= k_samples
            num_samples_clients[curr_cid] -= k_samples

            client_indices[curr_cid].extend(list(random_sample_idx))
            data_class_idx[curr_class] -= set(random_sample_idx)
            break

    client_dict = [client_indices[cid] for cid in range(len(q_client))]
    return client_dict