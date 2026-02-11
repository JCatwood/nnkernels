import torch
import sys
import time
import json

from models import GPVecchia, MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from loss import NllLoss

torch.manual_seed(1)
# %% tuning parameters
d = 2  # locs are sampled from R^d
m = 30
use_NN_for_testing = True
use_NN_for_training = True # whether to use NN for training data selection. If False, random selection will be used. Note that using NN for training is only supported for datasets with fixed locations between replicates.
cond_on_train = False
n_replicates_for_training = 'all'  # can be 'all' or a positive integer specifying the number of replicates to use for training
if len(sys.argv) > 3:
    kernel_train_name = sys.argv[1]
    assert kernel_train_name in (
        "MyMaternKernel",
        "MyNSKernel_Scale",
        "MyNSKernel_Lengthscale",
    ), "Invalid kernel_train_name (second) arguments"
    train_type = sys.argv[2]
    assert train_type in ("data", "simulation"), "Invalid train_type (third) argument"
    if train_type == "simulation":
        kernel_gen_name = sys.argv[3]
        assert kernel_gen_name in (
            "MyMaternKernel",
            "MyNSKernel_Scale",
            "MyNSKernel_Lengthscale",
        ), "Invalid kernel_gen_name (fourth) arguments"
    else:
        data_name = sys.argv[3]
else:
    train_type = "data"  # ["simulation", "data"]
    data_name = f"GP_d{d}_fixedlocs_mean0_NS_range_80_20"
    kernel_gen_name = "MyNSKernel_Scale"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
    kernel_train_name = "MyMaternKernel"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]

# %% model parameters
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    n_batch = 2048
    n_epoch = 30001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
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
    dataloader = Vecc_Dataloader_GP_sim(KernelClass, kernel_parms_init, d, "y")
elif train_type == "data":
    dataloader = Vecc_Dataloader_Dataset(data_name)
else:
    raise Exception("Unexpected train_type")
# %% initialize model
if kernel_train_name == "MyMaternKernel":
    kernel_parms_init = [0.5, 0.1, 1.5, 0.01]
    KernelClass = MyMaternKernel
elif kernel_train_name == "MyNSKernel_Scale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 0.3, 1.5, 0.01]
    KernelClass = MyNSKernel_Scale
elif kernel_train_name == "MyNSKernel_Lengthscale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 2.0, 0.01]
    KernelClass = MyNSKernel_Lengthscale
model = GPVecchia(KernelClass, *kernel_parms_init)
model.to(device)

# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001


# %% loss func
loss_function = NllLoss()
# %% model training
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(
            size=n_batch, m=m, n_replicates=n_replicates_for_training, use_NN=use_NN_for_training
    )
        X_batch, y_batch, y_true = (
            X_batch.to(device),
            y_batch.to(device),
            y_true.to(device),
        )
    # predict the target
    optimizer.zero_grad()
    y_pred, y_stderr = model(X_batch, y_batch, length=length)
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
# %% evaluate
model.to("cpu")
model.eval()
loss_MSE = torch.nn.MSELoss()
loss_NLL = NllLoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size=n_batch)
    else:
        X_batch_list = []
        y_batch_list = []
        y_true_list = []
        length_list = []
        for seed in range(dataloader.N_test):
            X_batch, y_batch, y_true, length = dataloader.get_test_batch(
                seed=seed, use_NN=use_NN_for_testing, cond_on_train=cond_on_train
                )
            X_batch_list.append(X_batch)
            y_batch_list.append(y_batch)
            y_true_list.append(y_true)
            length_list.append(length)
        X_batch = torch.cat(X_batch_list, dim=0)
        y_batch = torch.cat(y_batch_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        length = torch.cat(length_list, dim=0)
    y_pred, y_stderr = model(X_batch, y_batch, length=length)
    loss_NLL_val = loss_NLL(y_pred, y_true, y_stderr)
    loss_MSE_val = loss_MSE(y_pred, y_true)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "GPVecchia",
            "m": m,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": "GPVecchia",
            "m": m,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
