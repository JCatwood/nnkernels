import torch
import gpytorch
import os
import sys
import time
import glob
import re

from data_sim import prepare_sequence_cond_sd
from models import NNDT_Sum_NNTG, LSTM_NNTG, Kernel_Nugg

torch.manual_seed(1)

d = 2 # locs are sampled from R^d

if len(sys.argv) > 1:
    n_layer_DT = int(sys.argv[1])
    n_layer_TG = int(sys.argv[2])
    size_all = [int(sys.argv[i]) for i in range(3, len(sys.argv))]
    assert len(size_all) == n_layer_DT + n_layer_TG
    assert size_all[n_layer_DT - 1] == size_all[n_layer_DT]
    size_ST = [d] + size_all[:n_layer_DT]
    size_TG = size_all[n_layer_DT:] + [1]
else:
    n_layer_DT = 3
    n_layer_TG = 3
    # the latent dim is 3
    size_ST = [d, 64, 64, 3] 
    size_TG = [3, 32, 32, 1]
n_epoch = 4000
n_batch = 1000

seq_func = prepare_sequence_cond_sd
fixed_len = False
model = NNDT_Sum_NNTG(size_ST, size_TG)

# scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.001
    return base_lr/(1+factor*epoch)
    # return 0.001

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
kernel_gpt = gpytorch.kernels.MaternKernel(1.5)
kernel_gpt.lengthscale = 0.3
kernel = Kernel_Nugg(kernel_gpt, 0.01)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
model.to(device)

model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    # simulate training data
    if fixed_len:
        length = 10
    else:
        length = torch.randint(1, 30, (n_batch,))
    with torch.no_grad():
        locs, cond_sd = seq_func(length, kernel, nbatch=n_batch, d=d)
        locs_at_length = locs[torch.arange(n_batch), length - 1, :] # n_batch X d
        locs_at_length_reshape = locs_at_length.reshape(n_batch, 1, d)
        locs = locs - locs_at_length_reshape
        locs, cond_sd = locs.to(device), cond_sd.to(device)
    # predict the target
    optimizer.zero_grad()
    if fixed_len:
        cond_sd_pred = model(locs)
    else:
        cond_sd_pred = model(locs, length)
    loss = loss_function(cond_sd_pred, cond_sd)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        print(f"Total variation of y is {cond_sd.var().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        crt_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {crt_lr}", flush=True)
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
print(f"Total variation of y is {cond_sd.var().item()}", flush=True)
timer_prev = timer
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)