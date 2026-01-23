import torch
import gpytorch
import sys
import time

from dataloader import Vecc_Dataloader_GP_sim
from models import NNDT_Sum_NNTG, MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 2 # locs are sampled from R^d
target = 'inv_chol' # ["cond_sd", "cond_mean", "inv_chol"]
fixed_len = True
input_trans_type = 'locs_diff'
aggregate_mtd = 'sum'
m = 30
if target == 'cond_mean':
    input_trans_type += '_and_y'
if target == "inv_chol":
    model_output = "vector"
nfeatures = input_transformed_dim(d, input_trans_type)
kernel_name = "MyMaternKernel" # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
# %% model parameters
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_ST = [nfeatures, 128, 128, 128, 16] 
    if model_output == "vector":
        size_TG = [size_ST[-1] * 2, 128, 128, 128, 1]
    else:
        size_TG = [size_ST[-1], 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 10001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_ST = [nfeatures, 64, 64, 8] 
    if model_output == "vector":
        size_TG = [size_ST[-1] * 2, 64, 64, 1]
    else:
        size_TG = [size_ST[-1], 64, 64, 1]
    n_batch = 1024
    n_epoch = 5001
# %% define covariance kernel
if kernel_name == "MyMaternKernel":
    kernel = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
elif kernel_name == "MyNSKernel_Scale":
    kernel = MyNSKernel_Scale(-0.5, -1.2, -1.44, 0.3, 1.5, 0.01)
elif kernel_name == "MyNSKernel_Lengthscale":
    kernel = MyNSKernel_Lengthscale(-0.5, -1.2, -1.44, 2.0, 0.01)
# %% dataloader
dataloader = Vecc_Dataloader_GP_sim(kernel, d, fixed_len, m + 1, target=target)
# %% initialize model
model = NNDT_Sum_NNTG(size_ST, size_TG, aggregate_mtd=aggregate_mtd, output=model_output)
model.to(device)
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr/(1+factor*epoch)
    # return 0.001
# %% loss func
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# %% model training
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    with torch.no_grad():
        locs_batch, y_batch, y, length = dataloader.get_minibatch(size=n_batch)
        if target == 'cond_mean':
            X_batch_trans = input_transform(locs_batch, y_batch, length, input_trans_type)
        else:
            X_batch_trans = input_transform(locs_batch, None, length, input_trans_type)
        X_batch_trans, y = X_batch_trans.to(device), y.to(device)
    # predict the target
    optimizer.zero_grad()
    if target == 'cond_sd':
        y_pred = torch.exp(model(X_batch_trans, length))
    else:
        y_pred = model(X_batch_trans, length)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        print(f"Total variation of y is {y.var().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        crt_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {crt_lr}", flush=True)
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
print(f"Total variation of y is {y.var().item()}", flush=True)
timer_prev = timer
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)