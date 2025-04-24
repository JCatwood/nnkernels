import torch
import gpytorch
import os
import sys
import time

from loss_func import prepare_sequence_cond_sd, prepare_sequence_cond_mean
from loss_func import loss_len10_cond_sd, loss_len1to30_cond_sd
from loss_func import loss_len10_cond_mean, loss_len1to30_cond_mean
from lstm import LSTMKernelSD, LSTMKernelMean

torch.manual_seed(1)

if len(sys.argv) > 1:
    n_hidden = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    n_batch = int(sys.argv[3])
else:
    n_hidden = 256
    n_epoch = 4000
    n_batch = 100
fixed_len = False
task_type = "cond_mean" 
if task_type == "cond_sd":
    model = LSTMKernelSD(2, n_hidden)
    seq_func = prepare_sequence_cond_sd
    if fixed_len:
        loss_func = loss_len10_cond_sd
        output_fn = f"trained_models/cond_sd_len10_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
    else:
        loss_func = loss_len1to30_cond_sd
        output_fn = f"trained_models/cond_sd_len1-30_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
else: # "cond_mean"
    model = LSTMKernelMean(3, n_hidden)
    seq_func = prepare_sequence_cond_mean
    if fixed_len:
        loss_func = loss_len10_cond_mean
        output_fn = f"trained_models/cond_mean_len10_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
    else:
        loss_func = loss_len1to30_cond_mean
        output_fn = f"trained_models/cond_mean_len1-30_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kernel = gpytorch.kernels.MaternKernel(1.5)

os.makedirs("trained_models", exist_ok=True)
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    optimizer.zero_grad()
    loss = loss_func(model, loss_function, kernel, n_batch)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev } seconds", flush=True)
torch.save(model.state_dict(), output_fn)

model.eval()
with torch.no_grad():
    if fixed_len:
        n_locs = 10
        X, y = seq_func(n_locs, kernel, n_batch)
        y_pred, _, _ = model(X, None, None)
        loss = loss_function(y_pred, y)
        print(f"At length {n_locs}, total variation is {y.var().item()}, MSE is {loss.item()}")
    else:
        n_locs = torch.randint(1, 30, (n_batch,))
        X, y = seq_func(n_locs, kernel, n_batch)
        y_pred, _, _ = model(X, None, None)
        loss = loss_function(y_pred, y)
        print(f"With n_loc in [{min(n_locs)}, {max(n_locs)}], "
              f"total variation is {y.var().item()}, MSE is {loss.item()}")