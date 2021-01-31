import torch

data_dir = './data/Seinfeld_Scripts.txt'
seq_length = 16
batch_size = 512
num_epochs = 200
learning_rate = 0.001
embedding_dim = 400
hidden_dim = 512
n_layers = 2
load_settings = 1
log = True
show_every_n_batches = 300
gpu_available = False


def get_device():
    train_on_gpu = False
    if  torch.cuda.is_available():
        train_on_gpu = True
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    return device, train_on_gpu
