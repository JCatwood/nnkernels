import torch
import time
import json
import math
import random
import numpy as np
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from parse_args import parse_args
from models import init_DeepKernelNNGP, init_VGP

# %% varying tuning parameters
args = parse_args()
method = args["method"]
d = args["d"]
m = args["m"]
train_type = args["train_type"]
kernel_gen_name = args["kernel_gen_name"]
KernelGen = args["KernelGen"]
data_name = args["data_name"]
n_replicates = args["n_replicates"]

if n_replicates is not None and n_replicates > 1:
    data_seeds = range(n_replicates)
else:
    data_seeds = None

if torch.cuda.is_available():
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    n_batch = 2048
    n_iter = 60001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
    n_batch = 1024
    n_iter = 3001

if train_type == "simulation":
    if kernel_gen_name == "MyMaternKernel":
        kernel_gen_init = [1.0, [0.3 for _ in range(d)], 1.5, 0.01]
    elif kernel_gen_name == "MyNSKernel_Scale":
        kernel_gen_init = [0.25, -1.0, 0.25, 0.03, 0.5, 0.01]
    elif kernel_gen_name == "MyNSKernel_Lengthscale":
        kernel_gen_init = [-2.0, 1.0, -1.0, 1.0, 0.01]
    else:  # MyNSKernel_Kron
        kernel_gen_init = [0.0, -1.2, 1.2, 0.03, 0.5, 0.01]
    dataloader = Vecc_Dataloader_GP_sim(KernelGen, kernel_gen_init, d, "y", device=device)
else:
    dataloader = Vecc_Dataloader_Dataset(data_name, data_seeds)

def nll_loss(y_pred, y_true, y_stderr, eps=1e-6):
    y_stderr = y_stderr.clamp_min(eps)
    nll = 0.5 * ((y_true - y_pred) / y_stderr) ** 2 + \
        0.5 * math.log(2.0 * math.pi) + torch.log(y_stderr)
    return nll.mean()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(123)

# %% model init
if method == "DeepKernelNNGP":
    dropout_ratio = 0.2 if train_type == "data" else 0.0
    model, model_specs = init_DeepKernelNNGP(d, dropout=dropout_ratio, device=device)
elif method == "VGP":
    model, model_specs = init_VGP(d, device=device)
else:
    raise ValueError("Undefined method name")

# %% model training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
model.to(device)
model.train()

train_batch_size = n_batch
if train_type == "data":
    train_batch_size = min(n_batch, dataloader.offset_train[-1])

timer = time.perf_counter()
for iter in range(n_iter):
    X_batch, y_batch, y_true = dataloader.get_minibatch(size=train_batch_size, m=m)
    X_batch, y_batch, y_true = X_batch.to(device), y_batch.to(device), y_true.to(device)

    optimizer.zero_grad()
    y_pred, y_pred_stderr = model(X_batch, y_batch)
    loss = nll_loss(y_pred, y_true, y_pred_stderr)
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        print(f"Current LR: {optimizer.param_groups[0]['lr']}", flush=True)
        print(f"Loss after {iter} iterations is {loss.detach().item()}", flush=True)
# %% evaluate
model.to("cpu")
model.eval()
loss_MSE = torch.nn.MSELoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true = dataloader.get_test_batch(size=n_batch*10, m=m)
    else:
        X_batch, y_batch, y_true = dataloader.get_test_batch(size='all', m=m)
    y_pred, y_pred_stderr = model(X_batch, y_batch)
    loss_NLL_val = nll_loss(y_pred, y_true, y_pred_stderr)
    loss_MSE_val = loss_MSE(y_pred, y_true)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "d": d,
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": method,
            "m": m,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
            "model_specs": model_specs,
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": method,
            "m": m,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
            "n_replicates": n_replicates,
            "model_specs": model_specs,
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")

# %%
