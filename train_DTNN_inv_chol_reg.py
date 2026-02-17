import torch
import time
import sys
import json

from models import (
    GPVecchia,
    NNDT_Sum_NNTG,
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
use_NN_for_testing = False
use_NN_for_training = False 
n_replicates_for_training = 'all'  # can be 'all' or a positive integer 
cond_on_train = True 
prop_sim_data = 0.5  # proportion of training data in each batch that comes from the GP simulator (as opposed to the original data), only applicable when train_type is "data"
dropout_ratio = 0.5
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
    data_name = f"GP_d{d}_rndlocs_mean0_NS_range_2000_1000"
# %% model parameters
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_DT = [nfeatures, 128, 128, 128, 64] 
    size_TG_krig_coeff = [size_DT[-1] * 2, 128, 128, 128, 1]
    size_TG_cond_sd_inv = [size_DT[-1], 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 10001
    n_epoch_GP = 4001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_DT = [nfeatures, 64, 64, 16] 
    size_TG_krig_coeff = [size_DT[-1] * 2, 64, 64, 1]
    size_TG_cond_sd_inv = [size_DT[-1], 64, 64, 1]
    n_batch = 1024
    n_epoch = 4001
    n_epoch_GP = 4001
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
# %% initialize GP model as regularizer
model_GP = GPVecchia(MyMaternKernel, *[0.5, 0.1, 1.5, 0.01])
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001
# %% loss func
loss_function = NllLoss()
# %% GP training
optimizer = torch.optim.Adam(model_GP.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_GP.to(device)
model_GP.train()
timer = time.perf_counter()
for epoch in range(n_epoch_GP):
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
    y_pred, y_stderr = model_GP(X_batch, y_batch, length=length)
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
        print(f"Loss of GP after {epoch} iterations is {loss.detach().item()}", flush=True)
model_GP.to("cpu")
model_GP.eval()

# %% initialize NN models
model_krig_coeff = NNDT_Sum_NNTG(size_DT, size_TG_krig_coeff, concat_input=True, dropout=dropout_ratio)
model_cond_sd_inv = NNDT_Sum_NNTG(size_DT, size_TG_cond_sd_inv, concat_input=False, dropout=dropout_ratio)
model_krig_coeff.to(device)
model_cond_sd_inv.to(device)
# %% Create a dataloader using the trained GP model for regularization
dataloader_GP_reg = Vecc_Dataloader_GP_sim(
    MyMaternKernel, [0.5, 0.1, 1.5, 0.01], d, target="y"
    )
dataloader_GP_reg.kernel.load_state_dict(model_GP.kernel.state_dict())
# %% NN models training
optimizer = torch.optim.Adam(
    list(model_krig_coeff.parameters()) + list(model_cond_sd_inv.parameters()), lr=1
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_krig_coeff.train()
model_cond_sd_inv.train()
timer = time.perf_counter()
n_batch_sim = int(n_batch * prop_sim_data)
n_batch_data = n_batch - n_batch_sim
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch_data, y_batch_data, y_true_data, length_data = dataloader.get_minibatch(
            size=n_batch_data, m=m, n_replicates=n_replicates_for_training, 
            use_NN=use_NN_for_training
            )
        X_batch_sim, y_batch_sim, y_true_sim, length_sim = dataloader_GP_reg.get_minibatch(
            size=n_batch_sim, m=m
            )
        X_batch = torch.cat((X_batch_data, X_batch_sim), dim=0)
        y_batch = torch.cat((y_batch_data, y_batch_sim), dim=0)
        y_true = torch.cat((y_true_data, y_true_sim), dim=0)
        length = torch.cat((length_data, length_sim), dim=0)

        input = input_transform(X_batch, None, length, type=input_trans_type)
        input = input[:, :-1, :]
        y_batch = y_batch[:, :-1, :]
        input, y_batch, y_true = input.to(device), y_batch.to(device), y_true.to(device)
    # predict mean and stderr
    optimizer.zero_grad()
    krig_coeff = model_krig_coeff(input)
    y_pred = torch.sum(krig_coeff * y_batch, dim=1)
    y_stderr_inv = torch.exp(model_cond_sd_inv(input))
    loss = loss_function(y_pred, y_true, stderr_inv=y_stderr_inv)
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
model_krig_coeff.eval()
model_cond_sd_inv.eval()
model_krig_coeff.to("cpu")
model_cond_sd_inv.to("cpu")
loss_MSE = torch.nn.MSELoss()
loss_NLL = NllLoss()
with torch.no_grad():
    if train_type == "simulation":
        torch.manual_seed(123)
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size=n_batch*10, m=m)
    else:
        X_batch_list = []
        y_batch_list = []
        y_true_list = []
        length_list = []
        for seed in range(dataloader.N_test):
            X_batch, y_batch, y_true, length = dataloader.get_test_batch(
                seed=seed, m=m, use_NN=use_NN_for_testing, cond_on_train=cond_on_train
                )
            X_batch_list.append(X_batch)
            y_batch_list.append(y_batch)
            y_true_list.append(y_true)
            length_list.append(length)
        X_batch = torch.cat(X_batch_list, dim=0)
        y_batch = torch.cat(y_batch_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        length = torch.cat(length_list, dim=0)
    # GP prediction first
    y_pred_GP, y_stderr_GP = model_GP(X_batch, y_batch, length=length)
    loss_NLL_GP = loss_NLL(y_pred_GP, y_true, y_stderr_GP)
    loss_MSE_GP = loss_MSE(y_pred_GP, y_true)
    # transform input for the NN models
    input = input_transform(X_batch, None, length, type=input_trans_type)
    input = input[:, :-1, :]
    y_batch = y_batch[:, :-1, :]
    krig_coeff = model_krig_coeff(input)
    y_pred = torch.sum(krig_coeff * y_batch, dim=1)
    y_stderr_inv = torch.exp(model_cond_sd_inv(input))
    loss_NLL_val = loss_NLL(y_pred, y_true, stderr_inv=y_stderr_inv)
    loss_MSE_val = loss_MSE(y_pred, y_true)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "NN2",
            "m": m,
            "size_DT": size_DT,
            "size_TG_krig_coeff": size_TG_krig_coeff,
            "size_TG_cond_sd_inv": size_TG_cond_sd_inv,
            "transformation": input_trans_type,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
            "NLL_GP": loss_NLL_GP.item(),
            "MSE_GP": loss_MSE_GP.item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": "NN2",
            "m": m,
            "size_DT": size_DT,
            "size_TG_krig_coeff": size_TG_krig_coeff,
            "size_TG_cond_sd_inv": size_TG_cond_sd_inv,
            "transformation": input_trans_type,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
            "NLL_GP": loss_NLL_GP.item(),
            "MSE_GP": loss_MSE_GP.item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")

# %%
