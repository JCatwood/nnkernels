import torch
import time
import sys
import json

from models import (
    NNDT2_Sum_NNTG,
    MyMaternKernel,
    MyNSKernel_Scale,
    MyNSKernel_Lengthscale,
)
from loss import NllLoss, MyMSELoss
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 2  # locs are sampled from R^d
m = 30
input_trans_type = "dist_direction_lastloc"
nfeatures = input_transformed_dim(d, input_trans_type)
fixed_len = True # needs to be true for this experiment
pre_trained_mdl_fn = None
if len(sys.argv) > 2:
    train_type = sys.argv[1]
    assert train_type in ("data", "simulation"), "Invalid train_type (first) argument"
    if train_type == "simulation":
        kernel_gen_name = sys.argv[2]
        assert kernel_gen_name in (
            "MyMaternKernel",
            "MyNSKernel_Scale",
            "MyNSKernel_Lengthscale",
        ), "Invalid kernel_gen_name (third) arguments"
    else:
        data_name = sys.argv[2]
else:
    train_type = "data"  # ["simulation", "data"]
    kernel_gen_name = "MyNSKernel_Lengthscale"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
    data_name = "GP_d2_rndlocs_mean0_Matern_2000_1000"
# %% model parameters
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_DT1 = [nfeatures, 128, 128, 128, 16] 
    size_DT2 = [nfeatures, 128, 128, 128, 16] 
    size_TG = [size_DT1[-1] + size_DT2[-1], 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 10001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_DT1 = [nfeatures, 64, 64, 8] 
    size_DT2 = [nfeatures, 64, 64, 8] 
    size_TG = [size_DT1[-1] + size_DT2[-1], 64, 64, 1]
    n_batch = 1024
    n_epoch = 3001
# %% dataloader
if train_type == "simulation":
    # define covariance kernel used for generating data
    if kernel_gen_name == "MyMaternKernel":
        kernel_parms_init = [1.0, 0.3, 1.5, 0.01]
        KernelClass = MyMaternKernel
    elif kernel_gen_name == "MyNSKernel_Scale":
        kernel_parms_init = [-0.5, -1.2, -1.44, 0.3, 1.5, 0.01]
        KernelClass = MyNSKernel_Scale
    elif kernel_gen_name == "MyNSKernel_Lengthscale":
        kernel_parms_init = [-0.5, -1.2, -1.44, 2.0, 0.01]
        KernelClass = MyNSKernel_Lengthscale
    dataloader = Vecc_Dataloader_GP_sim(KernelClass, kernel_parms_init, d, fixed_len, m + 1, "y")
elif train_type == "data":
    dataloader = Vecc_Dataloader_Dataset(data_name)
else:
    raise Exception("Unexpected train_type")
# %% initialize model
if pre_trained_mdl_fn is None:
    model = NNDT2_Sum_NNTG(size_DT1, size_DT2, size_TG)
else:
    model = torch.load(pre_trained_mdl_fn, weights_only=False)
model.to(device)
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001


# %% loss func
loss_function = MyMSELoss()
# %% model training
optimizer = torch.optim.Adam(
    model.parameters(), lr=1
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(size=n_batch)
        input = input_transform(X_batch, None, length, type=input_trans_type)
        input = input[:, :-1, :]
        y_batch = y_batch[:, :-1, :]
        input, y_batch, y_true = input.to(device), y_batch.to(device), y_true.to(device)
    # predict mean and stderr
    optimizer.zero_grad()
    krig_coeff = model(input)
    y_pred = torch.sum(krig_coeff * y_batch, dim=1)
    loss = loss_function(y_pred, y_true, None)
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
# %% evaluate
model.to("cpu")
loss_MSE = torch.nn.MSELoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size=n_batch)
    else:
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(seed=0)
    input = input_transform(X_batch, None, length, type=input_trans_type)
    input = input[:, :-1, :]
    y_batch = y_batch[:, :-1, :]
    krig_coeff = model(input)
    y_pred = torch.sum(krig_coeff * y_batch, dim=1)
    loss = loss_function(y_pred, y_true, None)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "NN2",
            "m": m,
            "size_DT1": size_DT1,
            "size_DT2": size_DT2,
            "size_TG": size_TG,
            "transformation": input_trans_type,
            "same_length": fixed_len,
            "Loss": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": "NN2",
            "m": m,
            "size_DT1": size_DT1,
            "size_DT2": size_DT2,
            "size_TG": size_TG,
            "transformation": input_trans_type,
            "same_length": fixed_len,
            "Loss": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
