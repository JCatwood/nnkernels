import torch
import sys
import time
import glob
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

from read_data import read_data
from lstm import LSTMKernelSD, LSTMKernelMean

class TrainData(Dataset):
    def __init__(self, X, y, m=30, CondSetStrategy="vanilla_NN"):
        self.X = X
        self.y = y
        if CondSetStrategy == "vanilla_NN":
            knn = NearestNeighbors(n_neighbors=m+1)
            knn.fit(X)
            _, self.NN = knn.kneighbors(X)
        elif CondSetStrategy == "rand_NN":
            n = len(y)
            random_values = torch.rand(n, 2)
            knn = NearestNeighbors(n_neighbors=m+1)
            knn.fit(random_values)
            _, self.NN = knn.kneighbors(random_values)
        else:
            raise ValueError(f"Invalid CondSetStrategy = {CondSetStrategy}")
        # reverse such that 0 to n - 1 at the end
        self.NN = torch.tensor(self.NN[:, torch.arange(m, -1, -1)])
        if not torch.equal(self.NN[:, -1], torch.arange(0, len(y))):
            raise ValueError(f"Last col of NN is not 0 to n - 1, "
                             "probably due to overlapping locs in X")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, ind):
        return self.X[self.NN[ind]], self.y[self.NN[ind]]

class TestData(Dataset):
    def __init__(self, XTrain, yTrain, XTest, yTest, 
                 m=30, CondSetStrategy="vanilla_NN"):
        self.n_train = XTrain.size(0)
        self.n_test = XTest.size(0)
        self.X = torch.cat((XTrain, XTest), dim=0)
        self.y = torch.cat((yTrain, yTest), dim=0)
        if CondSetStrategy == "vanilla_NN":
            knn = NearestNeighbors(n_neighbors=m)
            knn.fit(XTrain)
            _, self.NN = knn.kneighbors(XTest)
            self.NN = torch.tensor(self.NN)
        elif CondSetStrategy == "rand_NN":
            self.NN = torch.randint(0, self.n_train, (self.n_test, m))
        else:
            raise ValueError(f"Invalid CondSetStrategy = {CondSetStrategy}")
        # add arange(n_train, n_train + n_test) to the last col
        self.NN = torch.cat((
            self.NN, 
            torch.arange(self.n_train, self.n_train + self.n_test).unsqueeze(-1)
            ), dim=-1)
        
    def __len__(self):
        return self.n_test
    
    def __getitem__(self, ind):
        return self.X[self.NN[ind]], self.y[self.NN[ind]]

if len(sys.argv) > 1:
    n_hidden = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    n_batch = int(sys.argv[3])
    data_name = sys.argv[4]
else:
    n_hidden = 256
    n_epoch = 30
    n_batch = 512
    data_name = "gp_matern15"

X_train, y_train = read_data(data_name, type='train')
d = X_train.size(1)
n_batch_fake = 0
my_dataset = TrainData(X_train, y_train.squeeze())
my_dataloader = DataLoader(my_dataset, batch_size=n_batch, shuffle=True)

# model initialization
mdl_mean = LSTMKernelMean(d + 1, n_hidden)
mdl_sd = LSTMKernelSD(d, n_hidden)
mdl_mean_output_fn = f"trained_models/mean_{data_name}_{n_hidden}_{n_epoch}.pt"
mdl_sd_output_fn = f"trained_models/sd_{data_name}_{n_hidden}_{n_epoch}.pt"
mdl_mean_pretrain_fn = glob.glob(
    f"trained_models/cond_mean_len1-30_LSTM_{n_hidden}_*.pt")
mdl_sd_pretrain_fn = glob.glob(
    f"trained_models/cond_sd_len1-30_LSTM_{n_hidden}_*.pt")
# load pre-trained model if any
if len(mdl_mean_pretrain_fn) > 0:
    pre_train_sz = [int(re.search(r'LSTM_\d+_\d+', fn).group().split('_')[2]) 
                    for fn in mdl_mean_pretrain_fn]
    ind_max_sz = max(enumerate(pre_train_sz), key=lambda x: x[1])[0]
    pre_train_fn = mdl_mean_pretrain_fn[ind_max_sz]
    pre_train_sz = int(re.search(r'LSTM_\d+_\d+', pre_train_fn).group().split('_')[2])
    mdl_mean.load_state_dict(torch.load(pre_train_fn, map_location=torch.device('cpu')))
    print(f"Loaded pre-trained model {pre_train_fn}", flush=True)
if len(mdl_sd_pretrain_fn) > 0:
    pre_train_sz = [int(re.search(r'LSTM_\d+_\d+', fn).group().split('_')[2]) 
                    for fn in mdl_sd_pretrain_fn]
    ind_max_sz = max(enumerate(pre_train_sz), key=lambda x: x[1])[0]
    pre_train_fn = mdl_sd_pretrain_fn[ind_max_sz]
    pre_train_sz = int(re.search(r'LSTM_\d+_\d+', pre_train_fn).group().split('_')[2])
    mdl_sd.load_state_dict(torch.load(pre_train_fn, map_location=torch.device('cpu')))
    print(f"Loaded pre-trained model {pre_train_fn}", flush=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
mdl_mean.to(device)
mdl_sd.to(device)

# model training
optimizer = torch.optim.Adam(list(mdl_mean.parameters()) + list(mdl_sd.parameters()), 
                             lr=0.001)
nll_constant = 0.5 * torch.log(torch.tensor(2 * torch.pi))
mdl_mean.train()
mdl_sd.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    nll_total = torch.tensor(0.0)
    for X_batch, y_batch in my_dataloader:  # check if shuffle messes up with X and NN: no
        with torch.no_grad():
            X_and_y_batch = torch.cat((X_batch, y_batch.unsqueeze(-1)), -1)
            X_and_y_batch[:, -1, -1] = 0  # check if y_batch is changed: no
            y_pred = y_batch[:, -1]
        X_and_y_batch.to(device)
        X_batch.to(device)
        y_pred.to(device)
        optimizer.zero_grad()
        y_mean_pred, _, _ = mdl_mean(X_and_y_batch, None, None)
        y_sd_pred, _, _ = mdl_sd(X_batch, None, None)
        nll_without_const = torch.log(y_sd_pred) + \
            0.5 * torch.pow((y_mean_pred - y_pred) / y_sd_pred, 2)
        nll_avg = nll_without_const.mean()
        nll_avg.backward()
        optimizer.step()
        with torch.no_grad():
            nll_avg.to('cpu')
            nll_total = nll_total + (nll_avg + nll_constant) * y_batch.size(0)
    if epoch % 10 == 0:
        print(f"NLL after {epoch} iterations is {nll_total.detach().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev } seconds", flush=True)

mdl_mean.eval()
mdl_sd.eval()
with torch.no_grad():
    X_test, y_test = read_data(data_name, type='test')
    my_dataset = TestData(X_train, y_train.squeeze(), X_test, y_test.squeeze())
    my_dataloader = DataLoader(my_dataset, batch_size=n_batch, shuffle=True)
    nll_total = torch.tensor(0.0)
    for X_batch, y_batch in my_dataloader:
        X_and_y_batch = torch.cat((X_batch, y_batch.unsqueeze(-1)), -1)
        X_and_y_batch[:, -1, -1] = 0 
        y_pred = y_batch[:, -1]
        X_and_y_batch.to(device)
        X_batch.to(device)
        y_pred.to(device)
        y_mean_pred, _, _ = mdl_mean(X_and_y_batch, None, None)
        y_sd_pred, _, _ = mdl_sd(X_batch, None, None)
        nll_without_const = torch.log(y_sd_pred) + \
            0.5 * torch.pow((y_mean_pred - y_pred) / y_sd_pred, 2)
        nll_without_const.to('cpu')
        nll_total = nll_total + nll_without_const.sum() + nll_constant * y_batch.size(0)
    print(f"NLL for the testing dataset is {nll_total.detach().item()}", flush=True)
