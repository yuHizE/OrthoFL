## Libraries
To install the required libraries, run:
```
pip install -r requirements.txt
```

## Dataset Preparation
- Create a data directory (e.g. `data/` in the project root) or set a custom directory with `args.data_dir`.
- Download the following datasets and save them in the `data/` folder.

### CIFAR-10/CIFAR-100
- CIFAR-10: [Download here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- CIFAR-100: [Download here](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- Unzip the `tar.gz` files, and save the extracted files into `data/cifar10/` and `data/cifar100` respectively, e.g., `data/cifar10/cifar-10-batches-py/`.

### MNIST
- MNIST: [Download here](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
- After downloading, extract the files (e.g. `.idx1-ubyte`) and store them in `data/mnist/MNIST` folder.

### 20 Newsgroups
- This dataset can be directly loaded via the `datasets` library.

### HAR
- HAR Dataset: [Download here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
- Unzip `UCI HAR Dataset.zip` and rename the folder to `har` for consistency.

## How to Run

### Launch the Server
1. Launch the server to await requests from clients. 
- Specify the GPU by `CUDA_VISIBLE_DEVICES=gpu_id`, server port by `--port` (e.g., `12345`), task by `-t`, and the random seed by `--seed`.
- Example command for running `cifar10` task:
```bash
CUDA_VISIBLE_DEVICES=0 python run_server.py -t cifar10 --port 12345 --seed 42
```

### Start the Clients
1. Identify Client IP addresses:
- Use this command to retrieve the machine's IP address:
```bash
hostname -I
```
2. Lauch the client to connect to the server and start training:
- The following command simulates running multiple clients on the same cluster by specifying `--n_clients N` to indicate the number of clients. 
- Replace the `-s xxx.xxx.xx.xx` with the actual client IP address.
- Specify the client port by `--port` (`e.g., 8765`). It will automatically assign ports, incrementing by `N`.
- Ensure the server port matches the port (e.g., `12345`) specified in the server command.
- Example for 10 clients:
```bash
CUDA_VISIBLE_DEVICES=1 python run_multi_clients.py --server_port 12345 --port 8461 --n_clients 10 -s xxx.xxx.xx.xx
```
