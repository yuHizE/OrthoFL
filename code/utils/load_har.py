import os
import numpy as np
import re
import pandas as pd
import torch

from torch.utils.data import Dataset


def collate_fn(batch):
    batch_data, batch_labels = zip(*batch)

    batch_data = torch.FloatTensor(batch_data)
    batch_labels = torch.LongTensor(batch_labels)

    return batch_data, batch_labels


def load_subject_data(data_dir, partition):
    subjects = []
    with open(os.path.join(data_dir, partition, f'subject_{partition}.txt'), 'r') as rf:
        for line in rf.readlines():
            subjects.append(int(line.strip()))
    subjects = np.array(subjects)

    X = []
    with open(os.path.join(data_dir, partition, f'X_{partition}.txt'), 'r') as rf:
        for line in rf.readlines():
            # print(len(re.split('\s+', line.strip())))
            X.append([float(x) for x in re.split('\s+', line.strip())])
    X = np.array(X)

    y = []
    with open(os.path.join(data_dir, partition, f'y_{partition}.txt'), 'r') as rf:
        for line in rf.readlines():
            y.append(int(line.strip()))
    y = np.array(y)

    return subjects, X, y


def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

def load_group(data_dir, filenames):
    loaded = list()
    for name in filenames:
        data = load_file(os.path.join(data_dir, name))
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)

    return loaded

def load_dataset_group(data_dir, partition):
    filepath = os.path.join(data_dir, partition, 'Inertial Signals')
    # load all 9 files as a single array
    filenames = []
    # total acceleration
    filenames += [f'total_acc_x_{partition}.txt', f'total_acc_y_{partition}.txt', f'total_acc_z_{partition}.txt']
    # body acceleration
    filenames += [f'body_acc_x_{partition}.txt', f'body_acc_y_{partition}.txt', f'body_acc_z_{partition}.txt']
    # body gyroscope
    filenames += [f'body_gyro_x_{partition}.txt', f'body_gyro_y_{partition}.txt', f'body_gyro_z_{partition}.txt']
    # load input data
    X = load_group(filepath, filenames)
    # load class output
    y = load_file(os.path.join(data_dir, partition, f'y_{partition}.txt')).reshape(-1)
    # load subjects
    subjects = load_file(os.path.join(data_dir, partition, f'subject_{partition}.txt')).reshape(-1)

    return subjects, X, y

def load_data(data_dir):
    # todo: make alpha
    _, X_test, Y_test = load_dataset_group(data_dir, 'test')
    Y_test = Y_test - 1
    testData = MyDataset(X_test, Y_test)

    trainData = []
    subject_train, X_train, Y_train = load_dataset_group(data_dir, 'train')

    Y_train = Y_train - 1

    for i in np.unique(subject_train):
        idx = np.where(subject_train == i)[0]
        trainData.append(MyDataset(X_train[idx], Y_train[idx]))

    return trainData, testData


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
