import torch
import json
import math
import random
import numpy as np
from dataloader import Vecc_Dataloader_GP_sim
import DeepKernelNNGP
from kernel_config import sim_kernel_config

# parse args 
d = 3
m = 30

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

def nll_loss(y_pred, y_true, y_stderr, eps=1e-6):
    y_stderr = y_stderr.clamp_min(eps)
    nll = 0.5 * ((y_true - y_pred) / y_stderr) ** 2 + \
        0.5 * math.log(2.0 * math.pi) + torch.log(y_stderr)
    return nll.mean()
loss_MSE = torch.nn.MSELoss()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

kernel_config = sim_kernel_config(d)

for kernel_name in kernel_config:
    set_seed(123)
    dataloader = Vecc_Dataloader_GP_sim(
        kernel_config[kernel_name]["class"], kernel_config[kernel_name]["init"], 
        d, "y", device=device)
    model = DeepKernelNNGP.GPVecchia(
        DeepKernelNNGP.ZeroMean, kernel_config[kernel_name]["class"], 
        [], kernel_config[kernel_name]["init"])
    model_specs = {
        'MeanClass': "ZeroMean",
        'mean_class_init': [],
        'CovClass': kernel_config[kernel_name]["class"].__name__,
        'cov_class_init': kernel_config[kernel_name]["init"],
    }
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_batch, y_batch, y_true = dataloader.get_test_batch(size=n_batch*10, m=m)
        X_batch, y_batch, y_true = X_batch.to(device), y_batch.to(device), y_true.to(device)
        y_pred, y_pred_stderr = model(X_batch, y_batch)
        loss_NLL_val = nll_loss(y_pred, y_true, y_pred_stderr)
        loss_MSE_val = loss_MSE(y_pred, y_true)
    output_dict = {
        "d": d,
        "data_type": "simulation",
        "kernel_sim": kernel_name,
        "model": "VGP_true",
        "m": m,
        "NLL": loss_NLL_val.item(),
        "MSE": loss_MSE_val.item(),
        "model_specs": model_specs,
        "time_total": 0,
    }
    print(">>>")
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
# %%
