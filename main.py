import torch
import time
import json
import math
import random
import numpy as np
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from parse_args import parse_args
from models import *

# %% varying tuning parameters
args = parse_args()
method = args["method"]
d = args["d"]
m = args["m"]
seed = args["seed"]
train_type = args["train_type"]
kernel_gen_name = args["kernel_gen_name"]
kernel_gen_init = args["kernel_gen_init"]
KernelGen = args["KernelGen"]
data_name = args["data_name"]
n_replicates = args["n_replicates"]
enforce_cross_group_nn = args["enforce_cross_group_nn"]
group_ind_col = args["group_ind_col"]
max_nobs_per_group = args["max_nobs_per_group"]
lengthscale_init = args["lengthscale_init"]

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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

if train_type == "simulation":
    dataloader = Vecc_Dataloader_GP_sim(KernelGen, kernel_gen_init, d, "y", device=device)
else:
    dataloader = Vecc_Dataloader_Dataset(data_name, data_seeds, enforce_cross_group_nn,
                                         group_ind_col, max_nobs_per_group, lengthscale_init)

# %% loss functions
def score_mean_se(scores):
    """Return the sample mean and its standard error.

    The standard error is computed as

        scores.std(unbiased=True) / sqrt(n).

    For a single observation, the standard error is returned as NaN.
    """
    scores = scores.reshape(-1)
    n = scores.numel()

    score_mean = scores.mean()

    if n <= 1:
        score_se = torch.full_like(score_mean, float("nan"))
    else:
        score_se = scores.std(unbiased=True) / math.sqrt(n)
    return score_mean, score_se
    
def nll_loss(y_pred, y_true, y_stderr, eps=1e-6):
    y_stderr = y_stderr.clamp_min(eps)

    nll = (
        0.5 * ((y_true - y_pred) / y_stderr) ** 2
        + 0.5 * math.log(2.0 * math.pi)
        + torch.log(y_stderr)
    )
    return score_mean_se(nll)

def gaussian_crps(y_pred, y_true, y_stderr, eps=1e-6):
    y_stderr = y_stderr.clamp_min(eps)

    z = (y_true - y_pred) / y_stderr

    standard_normal_pdf = (
        torch.exp(-0.5 * z ** 2)
        / math.sqrt(2.0 * math.pi)
    )

    standard_normal_cdf = (
        0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    )

    crps = y_stderr * (
        z * (2.0 * standard_normal_cdf - 1.0)
        + 2.0 * standard_normal_pdf
        - 1.0 / math.sqrt(math.pi)
    )
    return score_mean_se(crps)

def mse_loss(y_pred, y_true):
    squared_error = (y_pred - y_true) ** 2
    return score_mean_se(squared_error)

def coverage_95(y_pred, y_true, y_stderr, eps=1e-6):
    y_stderr = y_stderr.clamp_min(eps)
    lower = y_pred - 1.96 * y_stderr
    upper = y_pred + 1.96 * y_stderr
    covered = ((y_true >= lower) &(y_true <= upper)).to(y_pred.dtype)
    coverage = covered.mean()
    n = covered.numel()
    if n <= 1:
        coverage_se = torch.full_like(coverage, float("nan"),)
    else:
        coverage_se = torch.sqrt(coverage * (1.0 - coverage) / n)

    return coverage, coverage_se

# %% model init
if train_type == "data":
    dropout_ratio = 0.3
    if data_name in {"Argo", "GHRSST"}:
        special_case = data_name
    else:
        special_case = None
else:
    dropout_ratio = 0.0
    special_case = None
if method == "DeepKernelNNGP":
    model, model_specs = init_DeepKernelNNGP(d, device=device, case=special_case, dropout=dropout_ratio)
elif method == "VGP":
    model, model_specs = init_VGP(d, device=device, case=special_case, dropout=dropout_ratio)
elif method == "VGP_SM":
    model, model_specs = init_VGP_SM(d, device=device, case=special_case, dropout=dropout_ratio)
elif method == "VGP_Wilson2015Deep":
    model, model_specs = init_VGP_Wilson2015Deep(d, device=device, case=special_case, dropout=dropout_ratio)
elif method == "SPGP":
    model, model_specs = init_SPGP(d, m=m, device=device, case=special_case, dropout=dropout_ratio)
else:
    raise ValueError("Undefined method name")

# %% model training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
if train_type == "data":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_iter,
        eta_min=1e-4,
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_iter,
        eta_min=1e-5,
    )
model.to(device)
model.train()

train_batch_size = n_batch
if train_type == "data":
    train_batch_size = min(n_batch, dataloader.offset_train[-1])

timer_bgn = time.perf_counter()
timer = time.perf_counter()
for iter in range(n_iter):
    X_batch, y_batch, y_true = dataloader.get_minibatch(size=train_batch_size, m=m)
    X_batch, y_batch, y_true = X_batch.to(device), y_batch.to(device), y_true.to(device)

    optimizer.zero_grad()
    y_pred, y_pred_stderr = model(X_batch, y_batch)
    loss, _ = nll_loss(y_pred, y_true, y_pred_stderr)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if iter % 1000 == 0:
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        print(f"Current LR: {optimizer.param_groups[0]['lr']}", flush=True)
        print(f"Loss after {iter} iterations is {loss.detach().item()}", flush=True)
    if time.perf_counter() - timer_bgn > 3600 * 8: # max runtime 8 hours
        print(f"Stopping early after {iter + 1} iterations due to time limit.", flush=True)
        break
timer_end = time.perf_counter()
time_total = timer_end - timer_bgn
# %% evaluate
model.eval()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true = dataloader.get_test_batch(size=n_batch*10, m=m)
        X_batch, y_batch, y_true = X_batch.to(device), y_batch.to(device), y_true.to(device)
        y_pred, y_pred_stderr = model(X_batch, y_batch)
        loss_NLL_val, loss_NLL_se = nll_loss(y_pred, y_true, y_pred_stderr,)
        loss_CRPS_val, loss_CRPS_se = gaussian_crps(y_pred, y_true, y_pred_stderr,)
        loss_MSE_val, loss_MSE_se = mse_loss(y_pred, y_true,)
        coverage_95_val, coverage_95_se = coverage_95(y_pred, y_true, y_pred_stderr,)
    else:
        loss_NLL_val_total = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        loss_CRPS_val_total = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        loss_MSE_val_total = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        coverage_95_val_total = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        n_total = 0
        for k in range(dataloader.n_replicates):
            X_batch, y_batch, y_true = dataloader.get_test_batch(rep_ind=[k,], m=m)
            X_batch, y_batch, y_true = X_batch.to(device), y_batch.to(device), y_true.to(device)
            y_pred, y_pred_stderr = model(X_batch, y_batch)
            loss_NLL_val_total += nll_loss(y_pred, y_true, y_pred_stderr)[0] * y_pred.size(0)
            loss_CRPS_val_total += gaussian_crps(y_pred, y_true, y_pred_stderr)[0] * y_pred.size(0)
            loss_MSE_val_total += mse_loss(y_pred, y_true)[0] * y_pred.size(0)
            coverage_95_val_total += coverage_95(y_pred, y_true, y_pred_stderr)[0] * y_pred.size(0)
            n_total += y_pred.size(0)
        loss_NLL_val = loss_NLL_val_total / n_total
        loss_CRPS_val = loss_CRPS_val_total / n_total
        loss_MSE_val = loss_MSE_val_total / n_total
        coverage_95_val = coverage_95_val_total / n_total
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "d": d,
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": method,
            "m": m,
            "NLL": loss_NLL_val.item(),
            "NLL_SE": loss_NLL_se.item(),
            "CRPS": loss_CRPS_val.item(),
            "CRPS_SE": loss_CRPS_se.item(),
            "MSE": loss_MSE_val.item(),
            "MSE_SE": loss_MSE_se.item(),
            "Coverage95": 100.0 * coverage_95_val.item(),
            "Coverage95_SE": 100.0 * coverage_95_se.item(),
            "model_specs": model_specs,
            "time_total": time_total,
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": method,
            "m": m,
            "seed": seed,
            "NLL": loss_NLL_val.item(),
            "CRPS": loss_CRPS_val.item(),
            "MSE": loss_MSE_val.item(),
            "Coverage95": 100.0 * coverage_95_val.item(),
            "n_replicates": n_replicates,
            "model_specs": model_specs,
            "time_total": time_total,
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")

# %%
