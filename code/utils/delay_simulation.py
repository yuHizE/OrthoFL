# simulate the delays
# the order of aggregation is pre-calculated and fixed based on the delay of each devices.
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import heapq

MAX_TIME = float('inf')


def get_comm_info(comm_file):
    comm_delay = pd.read_csv(comm_file, header=None)
    comm_delay = comm_delay[~(comm_delay == 0).all(axis=1)].to_numpy()

    return comm_delay


def get_comp_info(comp_folder):
    comp_delay = []
    round_delay = []
    for file in os.listdir(comp_folder):
        if "comp_delay" in file:
            comp_delay.append(pd.read_csv(comp_folder+'/'+file,sep=",",header=None))
        if "round_delay" in file:
            round_delay.append(pd.read_csv(comp_folder+'/'+file,sep=",",header=None))
    comp_delay, round_delay = pd.concat(comp_delay),pd.concat(round_delay)

    other_delay = round_delay - comp_delay
    clients_comp_delay_std = []
    clients_comp_delay_mean = []
    clients_round_delay_std = []
    clients_round_delay_mean = []
    clients_other_delay_std = []
    clients_other_delay_mean = []
    for i in range(40):
        clients_comp_delay_std.append(np.std(comp_delay[i].to_numpy()[comp_delay[i].to_numpy().nonzero()]))
        clients_comp_delay_mean.append(np.mean(comp_delay[i].to_numpy()[comp_delay[i].to_numpy().nonzero()]))
        clients_round_delay_std.append(np.std(round_delay[i].to_numpy()[round_delay[i].to_numpy().nonzero()]))
        clients_round_delay_mean.append(np.mean(round_delay[i].to_numpy()[round_delay[i].to_numpy().nonzero()]))
        clients_other_delay_std.append(np.std(other_delay[i].to_numpy()[other_delay[i].to_numpy().nonzero()]))
        clients_other_delay_mean.append(np.mean(other_delay[i].to_numpy()[other_delay[i].to_numpy().nonzero()]))

    info_df = pd.DataFrame(clients_comp_delay_std)
    info_df[1] = clients_comp_delay_mean
    info_df[2] = clients_other_delay_std
    info_df[3] = clients_other_delay_mean
    info_df[4] = clients_round_delay_std
    info_df[5] = clients_round_delay_mean

    info_df = info_df.rename(columns={0: "comp_std", 1: "comp_mean", 2: "other_std", 3: "other_mean", 4: "round_std", 5: "round_mean"})

    info_df = info_df.fillna(0)

    # 0 6 7 9 -> 10 11 12 13
    info_df.iloc[0] = info_df.iloc[10]
    info_df.iloc[6] = info_df.iloc[11]
    info_df.iloc[7] = info_df.iloc[12]
    info_df.iloc[9] = info_df.iloc[13]

    info_df = info_df.iloc[[0, 1, 2, 3, 4, 5, 6, 8, 9]]

    return info_df


def get_delay(n_clients, comp_folder='delays/Archive/fedml_fedasync_mnist_noniid_g3_c40_c7_coreset_gurobi_ds600_sgd_0.01_e5_r40_gr10_ar2_0', comm_file='delays/delay_client_to_gateway.csv', round_delay_from_raspberrypi=True):
    comp_info = get_comp_info(comp_folder)

    if round_delay_from_raspberrypi:
        delay_mean, delay_std = comp_info['round_mean'].to_numpy(), comp_info['round_std'].to_numpy()
        if n_clients < len(delay_mean):
            selected_idx = np.random.choice(range(len(delay_mean)), n_clients, replace=False)
            delay_mean, delay_std = delay_mean[selected_idx], delay_std[selected_idx]
        else:
            selected_idx = np.random.choice(range(len(delay_mean)), n_clients)
            delay_mean, delay_std = delay_mean[selected_idx], delay_std[selected_idx]
    else:
        comm_info = get_comm_info(comm_file)
        comp_delay_mean, comp_delay_std = comp_info['comp_mean'].to_numpy(), comp_info['comp_std'].to_numpy()

        if n_clients < len(comp_delay_mean):
            comp_selected_idx = np.random.choice(range(len(comp_delay_mean)), n_clients, replace=False)
            comm_selected_idx = np.random.choice(range(len(comm_info)), n_clients, replace=False)
        else:
            comp_selected_idx = np.random.choice(range(len(comp_delay_mean)), n_clients)
            comm_selected_idx = np.random.choice(range(len(comm_info)), n_clients)

        comp_delay_mean, comp_delay_std = comp_delay_mean[comp_selected_idx], comp_delay_std[comp_selected_idx]
        comm_delay_mean = np.array([np.random.choice(d[d != 0]) for d in comm_info[comm_selected_idx]])
        delay_mean = comp_delay_mean + comm_delay_mean
        delay_std = comp_delay_std

    return delay_mean, delay_std


def get_update_sequence(n_updates, delay_mean, delay_std):
    num_clients = len(delay_mean)

    if np.all(delay_std == 0) and np.all(delay_mean == delay_mean[0]):  # same delay time for all clients
        update_sequence = np.tile(np.random.permutation(num_clients), math.ceil(n_updates / num_clients))[:n_updates].tolist()
        update_timestamp = [delay_mean[0] * (i // num_clients) + 0.001 * (i % num_clients) for i in range(n_updates)]
        return update_sequence, update_timestamp

    client_update_time = np.empty((num_clients, n_updates))
    for i, (mean, std) in enumerate(zip(delay_mean, delay_std)):
        round_time = np.random.normal(mean, std, n_updates)
        round_time[round_time <= 0] = mean
        client_update_time[i, :] = np.cumsum(round_time)

    # Initialize the heap with the first update time of each client
    heap = [(client_update_time[i, 0], i, 0) for i in range(num_clients)]
    heapq.heapify(heap)

    update_sequence = []
    update_timestamp = []
    for _ in tqdm(range(n_updates), total=n_updates):
        # Extract the smallest update time
        min_time, client_id, index = heapq.heappop(heap)
        update_sequence.append(client_id)
        update_timestamp.append(min_time)

        # Push the next update time for this client into the heap
        if index + 1 < n_updates:
            next_time = client_update_time[client_id, index + 1]
            heapq.heappush(heap, (next_time, client_id, index + 1))
        else:
            heapq.heappush(heap, (float('inf'), client_id, index + 1))  # Effectively remove from consideration

    return update_sequence, update_timestamp


def get_client_update_time(n_updates, delay_mean, delay_std):
    client_update_time = []
    for mean, std in zip(delay_mean, delay_std):
        # generate round time with mean and std. there may be some perturbation in the delay time at each round
        round_time = np.random.normal(mean, std, n_updates)
        round_time[round_time <= 0] = mean
        # accumulate round time
        client_update_time.append(round_time) # no accumulation
    client_update_time = np.array(client_update_time)
    return client_update_time
