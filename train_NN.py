import torch
import time
import sys
import json

from models import (
    NNDT_Sum_NNTG,
    GPVecchia,
    MyMaternKernel,
    MyNSKernel_Scale,
    MyNSKernel_Lengthscale,
)
from loss import NllLoss
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 2  # locs are sampled from R^d
m = 30
dropout_ratio = 0.5
input_trans_type = "dist_direction_lastloc"
input_trans_mean = input_trans_type + "_and_y"
input_trans_sd = input_trans_type
nfeature_mean = input_transformed_dim(d, input_trans_mean)
nfeature_sd = input_transformed_dim(d, input_trans_sd)
if len(sys.argv) > 3:
    if sys.argv[1].lower().strip() in ("yes", "true", "t", "y", "1", "on"):
        fixed_len = True
    elif sys.argv[1].lower().strip() in ("no", "false", "f", "n", "0", "off"):
        fixed_len = False
    else:
        raise ValueError(f"Invalid boolean value: '{sys.argv[1]}'")
    train_type = sys.argv[2]
    assert train_type in ("data", "simulation"), "Invalid train_type (second) argument"
    if train_type == "simulation":
        kernel_gen_name = sys.argv[3]
        assert kernel_gen_name in (
            "MyMaternKernel",
            "MyNSKernel_Scale",
            "MyNSKernel_Lengthscale",
        ), "Invalid kernel_gen_name (third) arguments"
    else:
        data_name = sys.argv[3]
        if len(sys.argv) > 4:
            data_seed = int(sys.argv[4])
        else:
            data_seed = None
else:
    fixed_len = True
    train_type = "data"  # ["simulation", "data"]
    kernel_gen_name = "MyNSKernel_Scale"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
    data_name = "GP_NS_scale_2000_1000"
    data_seed = 0
# %% model parameters
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_DT_mean = [nfeature_mean, 128, 128, 128, 8]
    size_TG_mean = [8, 128, 128, 128, 1]
    size_DT_sd = [nfeature_sd, 128, 128, 128, 8]
    size_TG_sd = [8, 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 30000
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
    size_DT_mean = [nfeature_mean, 96, 96, 4]
    size_TG_mean = [4, 96, 96, 1]
    size_DT_sd = [nfeature_sd, 64, 64, 3]
    size_TG_sd = [3, 32, 32, 1]
    n_batch = 1024
    n_epoch = 4000
# %% dataloader
if train_type == "simulation":
    # define covariance kernel used for generating data
    if kernel_gen_name == "MyMaternKernel":
        kernel_gen = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
    elif kernel_gen_name == "MyNSKernel_Scale":
        kernel_gen = MyNSKernel_Scale(-0.5, -1.2, -1.44, 0.3, 1.5, 0.01)
    elif kernel_gen_name == "MyNSKernel_Lengthscale":
        kernel_gen = MyNSKernel_Lengthscale(-0.5, -1.2, -1.44, 2.0, 0.01)
    dataloader = Vecc_Dataloader_GP_sim(kernel_gen, d, fixed_len, m + 1, target="y")
elif train_type == "data":
    if data_seed is None:
        dataloader = Vecc_Dataloader_Dataset(data_name, fixed_len, length_max=m + 1)
    else:
        dataloader = Vecc_Dataloader_Dataset(
            data_name, fixed_len, length_max=m + 1, seed=data_seed
        )
else:
    raise Exception("Unexpected train_type")
# %% initialize model
model_mean = NNDT_Sum_NNTG(size_DT_mean, size_TG_mean, dropout=dropout_ratio)
model_sd = NNDT_Sum_NNTG(size_DT_sd, size_TG_sd, dropout=dropout_ratio)
model_mean.to(device)
model_sd.to(device)


# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001


# %% loss func
loss_function = NllLoss()
# %% model training
optimizer = torch.optim.Adam(
    [{"params": model_mean.parameters()}, {"params": model_sd.parameters()}], lr=1
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_mean.train()
model_sd.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(size=n_batch)
        input_mean = input_transform(X_batch, y_batch, length, type=input_trans_mean)
        input_sd = input_transform(X_batch, None, length, type=input_trans_sd)
        input_mean, input_sd, y_true = (
            input_mean.to(device),
            input_sd.to(device),
            y_true.to(device),
        )
    # predict mean and stderr
    optimizer.zero_grad()
    y_pred = model_mean(input_mean, length=length)
    y_stderr = torch.exp(model_sd(input_sd, length=length))
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
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
# %% evaluate
model_mean.to("cpu")
model_sd.to("cpu")
model_mean.eval()
model_sd.eval()
loss_MSE = torch.nn.MSELoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size=n_batch)
    else:
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(
            size=dataloader.n_test
        )
    input_mean = input_transform(X_batch, y_batch, length, type=input_trans_mean)
    input_sd = input_transform(X_batch, None, length, type=input_trans_sd)

    y_pred = model_mean(input_mean, length=length)
    y_stderr = torch.exp(model_sd(input_sd, length=length))
    loss = loss_function(y_pred, y_true, y_stderr)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "NN",
            "m": m,
            "dropout": dropout_ratio,
            "size_DT_mean": size_DT_mean,
            "size_TG_mean": size_TG_mean,
            "size_DT_sd": size_DT_sd,
            "size_TG_sd": size_TG_sd,
            "transformation": input_trans_type,
            "same_length": fixed_len,
            "NLL": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "seed": data_seed,
            "model": "NN",
            "m": m,
            "dropout": dropout_ratio,
            "size_DT_mean": size_DT_mean,
            "size_TG_mean": size_TG_mean,
            "size_DT_sd": size_DT_sd,
            "size_TG_sd": size_TG_sd,
            "transformation": input_trans_type,
            "same_length": fixed_len,
            "NLL": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
