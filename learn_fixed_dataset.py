import torch
import gpytorch
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

from models import NNDT_Sum_NNTG, GPVecchia, MyMaternKernel
from data_sim import sim_GP
from prepare_seq import prepare_seq
from loss import NllLoss
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)

d = 2 # locs are sampled from R^d
m = 30
n_train = 1000
n_test = 1000
n_epoch = 4000
n_batch = 1000
use_noise = False
dropout_ratio = 0.5
input_trans_mean = "locs_diff_and_y"
input_trans_sd = "locs_diff"
nfeature_mean = input_transformed_dim(d, input_trans_mean)
nfeature_sd = input_transformed_dim(d, input_trans_sd)
if use_noise:
    noise_factor = 0.05

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')

kernel = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
locs_train, locs_test, y_train, y_test = sim_GP(n_train, n_test, kernel, d)
if use_noise:
    locs_sd, y_sd = torch.std(locs_train, dim=0, keepdim=True), \
        torch.std(y_train, dim=0, keepdim=True)
    locs_sd, y_sd = locs_sd.to(device), y_sd.to(device)
NN_search_obj = NearestNeighbors(n_neighbors=m, algorithm='auto')
NN_search_obj.fit(locs_train)
NN_train = torch.from_numpy(NN_search_obj.kneighbors(locs_train, m+1, return_distance=False))
NN_test = torch.from_numpy(NN_search_obj.kneighbors(locs_test, m, return_distance=False))
NN_train_rev = NN_train[:, torch.arange(m, -1, -1)]
NN_test_rev = NN_test[:, torch.arange(m - 1, -1, -1)]

size_DT_mean = [nfeature_mean, 96, 96, 4] 
size_TG_mean = [4, 96, 96, 1]
size_DT_sd = [nfeature_sd, 64, 64, 3] 
size_TG_sd = [3, 32, 32, 1]
model_mean = NNDT_Sum_NNTG(size_DT_mean, size_TG_mean, dropout=dropout_ratio)
model_sd = NNDT_Sum_NNTG(size_DT_sd, size_TG_sd, dropout=dropout_ratio)
model_mean.to(device)
model_sd.to(device)

locs_batch_test = torch.cat((locs_train[NN_test_rev, :], locs_test.unsqueeze(1)), dim=1)
y_batch_test = torch.cat((y_train[NN_test_rev], torch.zeros(n_test, 1)), dim=-1)
input_mean_test = input_transform(locs_batch_test, y_batch_test.unsqueeze(-1), type=input_trans_mean)
input_sd_test = input_transform(locs_batch_test, None, type=input_trans_sd)
input_mean_test, input_sd_test, y_test = input_mean_test.to(device), input_sd_test.to(device), \
    y_test.to(device)

# scheduler
def lr_lambda(epoch):
    # base_lr = 0.001
    # factor = 0.001
    # return base_lr/(1+factor*epoch)
    return 0.001

loss_function = NllLoss()
optimizer = torch.optim.Adam([{'params': model_mean.parameters()}, 
                              {'params': model_sd.parameters()}], lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_mean.train()
model_sd.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    # draw mini-batches
    with torch.no_grad():
        locs_batch, y_batch = prepare_seq(locs_train, y_train, NN_train_rev, size=n_batch)
        if use_noise:
            locs_batch = locs_batch + \
                locs_sd.unsqueeze(0) * noise_factor * torch.randn_like(locs_batch)
            y_batch = y_batch + \
                y_sd.unsqueeze(0) * noise_factor * torch.randn_like(y_batch)
        y_true = y_batch[:, -1].clone()
        y_batch[torch.arange(n_batch), m] = 0.0
        input_mean = input_transform(locs_batch, y_batch.unsqueeze(-1), type=input_trans_mean)
        input_sd = input_transform(locs_batch, None, type=input_trans_sd)
        input_mean, input_sd, y_true = input_mean.to(device), \
            input_sd.to(device), y_true.to(device)
    # predict mean and stderr
    optimizer.zero_grad()
    y_pred = model_mean(input_mean, length=m+1)
    y_stderr = torch.exp(model_sd(input_sd, length=m+1))
    loss = loss_function(y_pred, y_true, y_stderr)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 1000 == 0:
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        crt_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {crt_lr}", flush=True)
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        with torch.no_grad():
            y_pred_test = model_mean(input_mean_test, length=m+1)
            y_stderr_test = torch.exp(model_sd(input_sd_test, length=m+1))
            loss_test = loss_function(y_pred_test, y_test, y_stderr_test)
            print(f"Loss of the testing dataset is {loss_test.detach().item()}", flush=True)
timer_prev = timer
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
model_mean.eval()
model_sd.eval()
with torch.no_grad():
    y_pred_test = model_mean(input_mean_test, length=m+1)
    y_stderr_test = torch.exp(model_sd(input_sd_test, length=m+1))
    loss_test = loss_function(y_pred_test, y_test, y_stderr_test)
    print(f"Loss of the testing dataset is {loss_test.detach().item()}", flush=True)

# compare with GPVecchia initiated with the true kernel
with torch.no_grad():
    mdl_GP = GPVecchia(MyMaternKernel, *[1.0, 0.3, 1.5, 0.01])
    y_pred_test_GP, y_stderr_test_GP = mdl_GP(locs_batch_test, y_batch_test)
    loss_test_GP = loss_function(y_pred_test_GP, y_test, y_stderr_test_GP)
    print(f"Loss of GP using the testing dataset is {loss_test_GP.detach().item()}", flush=True)

    mdl_GP = GPVecchia(MyMaternKernel, *[1.0, 0.1, 0.5, 0.005])
    y_pred_test_GP, y_stderr_test_GP = mdl_GP(locs_batch_test, y_batch_test)
    loss_test_GP = loss_function(y_pred_test_GP, y_test, y_stderr_test_GP)
    print(f"Loss of mis-specified GP using the testing dataset is {loss_test_GP.detach().item()}", flush=True)