import torch
import sys
import time
import json
import copy

from models import (
    NNDT_Sum_NNTG,
    GPVecchia,
    MyMaternKernel,
    MyNSKernel_Scale,
    MyNSKernel_Lengthscale,
)
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from loss import NllLoss, MyMSELoss
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 2  # locs are sampled from R^d
m = 30
input_trans_type = "dist_direction_lastloc"
input_trans_mean = input_trans_type + "_and_y"
input_trans_sd = input_trans_type
nfeature_mean = input_transformed_dim(d, input_trans_mean)
nfeature_sd = input_transformed_dim(d, input_trans_sd)
train_type = "data"
fixed_len = True
if len(sys.argv) > 2:
    kernel_train_name = sys.argv[1]
    assert kernel_train_name in (
        "MyMaternKernel",
        "MyNSKernel_Scale",
        "MyNSKernel_Lengthscale",
    ), "Invalid kernel_train_name (second) arguments"
    data_name = sys.argv[2]
    if len(sys.argv) > 3:
        data_seed = int(sys.argv[3])
    else:
        data_seed = None
else:
    kernel_train_name = "MyMaternKernel"
    data_name = "GP_NS_range_2000_1000"
    data_seed = 0

# %% device and number of training iterations
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    n_batch = 2048
    n_epoch = 30001
    size_DT_mean = [nfeature_mean, 128, 128, 128, 8]
    size_TG_mean = [8, 128, 128, 128, 1]
    size_DT_sd = [nfeature_sd, 128, 128, 128, 8]
    size_TG_sd = [8, 128, 128, 128, 1]
    prop_sim_data_lst = [0.9, 0.8, 0.7, 0.6, 0.5]
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
    n_batch = 1024
    n_epoch = 4001
    size_DT_mean = [nfeature_mean, 96, 96, 4]
    size_TG_mean = [4, 96, 96, 1]
    size_DT_sd = [nfeature_sd, 64, 64, 3]
    size_TG_sd = [3, 32, 32, 1]
    prop_sim_data_lst = [0.9, 0.7, 0.5]

# %% dataloader
if data_seed is None:
    dataloader = Vecc_Dataloader_Dataset(data_name, fixed_len, length_max=m + 1)
else:
    dataloader = Vecc_Dataloader_Dataset(
        data_name, fixed_len, length_max=m + 1, seed=data_seed
    )
if n_batch > dataloader.n_train:
    n_batch = dataloader.n_train
# %% get test batch
with torch.no_grad():
    X_batch_test, y_batch_test, y_true_test, length_test = dataloader.get_test_batch(
        size=dataloader.n_test
    )
    X_batch_test, y_batch_test, y_true_test = (
        X_batch_test.to(device),
        y_batch_test.to(device),
        y_true_test.to(device),
    )

# %% scheduler and loss func
def lr_lambda(epoch):
    base_lr = 0.002
    factor = 0.0001
    return base_lr / (1 + factor * epoch)


loss_MSE = MyMSELoss()
loss_NLL = NllLoss()
loss_function = loss_MSE

# %% initialize GPVecchia
if kernel_train_name == "MyMaternKernel":
    kernel_parms_init = [0.5, 0.1, 1.5, 0.01]
    KernelClass = MyMaternKernel
elif kernel_train_name == "MyNSKernel_Scale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 0.3, 1.5, 0.01]
    KernelClass = MyNSKernel_Scale
elif kernel_train_name == "MyNSKernel_Lengthscale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 2.0, 0.01]
    KernelClass = MyNSKernel_Lengthscale
model_GPVecchia = GPVecchia(KernelClass, *kernel_parms_init)
model_GPVecchia.to(device)

# %% GPVecchia training
optimizer = torch.optim.Adam(model_GPVecchia.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_GPVecchia.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(size=n_batch)
        X_batch, y_batch, y_true = (
            X_batch.to(device),
            y_batch.to(device),
            y_true.to(device),
        )
    # predict the target
    optimizer.zero_grad()
    y_pred, y_stderr = model_GPVecchia(X_batch, y_batch, length=length)
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
        print(
            f"GPVecchia Loss after {epoch} iterations is {loss.detach().item()}",
            flush=True,
        )

# %% evaluate with testing data
model_GPVecchia.eval()
with torch.no_grad():
    y_pred, y_stderr = model_GPVecchia(X_batch_test, y_batch_test, length=length_test)
    nll = loss_function(y_pred, y_true_test, y_stderr)
    mse = loss_MSE(y_pred, y_true_test)
    print(
        f"For testing dataset, GPVecchia NLL {nll.detach().item()}, MSE {mse.detach().item()}",
        flush=True,
    )
model_GPVecchia.to('cpu')

# %% initialize NN
model_NN_mean = NNDT_Sum_NNTG(size_DT_mean, size_TG_mean)
model_NN_sd = NNDT_Sum_NNTG(size_DT_sd, size_TG_sd)
model_NN_mean.to(device)
model_NN_sd.to(device)

# %% model training
optimizer = torch.optim.Adam(
    [{"params": model_NN_mean.parameters()}, {"params": model_NN_sd.parameters()}], lr=1
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_NN_mean.train()
model_NN_sd.train()
timer = time.perf_counter()
dataloader_sim = Vecc_Dataloader_GP_sim(
    copy.deepcopy(model_GPVecchia.kernel), d, fixed_len, m + 1, target="y"
)
input_mean_test = input_transform(
    X_batch_test, y_batch_test, length_test, type=input_trans_mean
)
input_sd_test = input_transform(
    X_batch_test, None, length_test, type=input_trans_sd
)
for prop_sim_data in prop_sim_data_lst:
    n_batch_sim = int(n_batch * prop_sim_data)
    n_batch_data = n_batch - n_batch_sim
    for epoch in range(n_epoch):
        with torch.no_grad():
            X_batch_data, y_batch_data, y_true_data, length_data = (
                dataloader.get_minibatch(size=n_batch_data)
            )
            X_batch_sim, y_batch_sim, y_true_sim, length_sim = (
                dataloader_sim.get_minibatch(size=n_batch_sim)
            )
            X_batch = torch.cat((X_batch_data, X_batch_sim), dim=0)
            y_batch = torch.cat((y_batch_data, y_batch_sim), dim=0)
            y_true = torch.cat((y_true_data, y_true_sim), dim=0)
            length = torch.cat((length_data, length_sim), dim=0)
            input_mean = input_transform(
                X_batch, y_batch, length, type=input_trans_mean
            )
            input_sd = input_transform(X_batch, None, length, type=input_trans_sd)
            input_mean, input_sd, y_true = (
                input_mean.to(device),
                input_sd.to(device),
                y_true.to(device),
            )
        # predict mean and stderr
        optimizer.zero_grad()
        y_pred = model_NN_mean(input_mean, length=length)
        y_stderr = torch.exp(model_NN_sd(input_sd, length=length))
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
            print(
                f"Loss after {epoch} iterations is {loss.detach().item()}",
                flush=True,
            )
            with torch.no_grad():
                y_pred = model_NN_mean(input_mean_test, length=length_test)
                y_stderr = torch.exp(model_NN_sd(input_sd_test, length=length_test))
                nll = loss_NLL(y_pred, y_true_test, y_stderr)
                mse = loss_MSE(y_pred, y_true_test)
                print(
                    f"For testing dataset, NN NLL {nll.detach().item()}, MSE {mse.detach().item()}",
                    flush=True,
                )
    with torch.no_grad():
        y_pred = model_NN_mean(input_mean_test, length=length_test)
        y_stderr = torch.exp(model_NN_sd(input_sd_test, length=length_test))
        nll = loss_NLL(y_pred, y_true_test, y_stderr)
        mse = loss_MSE(y_pred, y_true_test)
    print(">>>")
    output_dict = {
        "data_type": train_type,
        "data_name": data_name,
        "seed": data_seed,
        "model": "GPVecchia_NN",
        "m": m,
        "same_length": fixed_len,
        "NLL": nll.item(),
        "MSE": mse.item(),
    }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
