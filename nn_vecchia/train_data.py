import torch
import sys
import time
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

from read_data import read_data
from lstm import LSTMKernelSD, LSTMKernelMean

class MyDataset(Dataset):
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
        self.NN = torch.tensor(self.NN[:, torch.arange(m, -1, -1)]  )
        if not torch.equal(self.NN[:, -1], torch.arange(0, len(y))):
            raise ValueError(f"Last col of NN is not 0 to n - 1, "
                             "probably due to overlapping locs in X")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, ind):
        return self.X[self.NN[ind]], self.y[self.NN[ind]]

if len(sys.argv) > 1:
    n_hidden = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    n_batch = int(sys.argv[3])
    data_name = sys.argv[4]
else:
    n_hidden = 256
    n_epoch = 4000
    n_batch = 128
    data_name = "gp_matern15"

X_train, y_train = read_data(data_name, type='train')
d = X_train.size(1)
my_dataset = MyDataset(X_train, y_train.squeeze())
my_dataloader = DataLoader(my_dataset, batch_size=1024, shuffle=True)
n_batch_fake = 0

# model initialization
mdl_mean = LSTMKernelMean(d + 1, n_hidden)
mdl_sd = LSTMKernelSD(d, n_hidden)
mdl_mean_pretrain_fn = glob.glob(
    f"trained_models/cond_mean_len1-30_LSTM_{n_hidden}_*.pt")
mdl_sd_pretrain_fn = glob.glob(
    f"trained_models/cond_sd_len1-30_LSTM_{n_hidden}_*.pt")
if len(mdl_mean_pretrain_fn) > 0:
    mdl_mean.load_state_dict(torch.load(mdl_mean_pretrain_fn[0]))
if len(mdl_sd_pretrain_fn) > 0:
    mdl_sd.load_state_dict(torch.load(mdl_sd_pretrain_fn[0]))

# model training
optimizer = torch.optim.Adam(list(mdl_mean.parameters()) + list(mdl_sd.parameters()), lr=0.001)
mdl_mean.train()
mdl_sd.train()
timer = time.perf_counter()
nll_constant = 0.5 * torch.log(torch.tensor(2 * torch.pi))
for epoch in range(n_epoch):
    nll_total = torch.tensor(0.0)
    for X_batch, y_batch in my_dataloader:  # check if shuffle messes up with X and NN: no
        with torch.no_grad():
            X_and_y_batch = torch.cat((X_batch, y_batch.unsqueeze(-1)), -1)
            X_and_y_batch[:, -1, -1] = 0  # check if y_batch is changed: no
            y_pred = y_batch[:, -1]
        optimizer.zero_grad()
        y_mean_pred, _, _ = mdl_mean(X_and_y_batch, None, None)
        y_sd_pred, _, _ = mdl_sd(X_batch, None, None)
        nll = nll_constant * y_batch.size(0) + torch.log(y_sd_pred) + \
            0.5 * torch.pow((y_mean_pred - y_pred) / y_sd_pred, 2)
        nll_sum = nll.sum()
        nll_sum.backward()
        optimizer.step()
        with torch.no_grad():
            nll_total = nll_total + nll_sum
    if epoch % 100 == 0:
        print(f"NLL after {epoch} iterations is {nll_total.detach().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev } seconds", flush=True)

