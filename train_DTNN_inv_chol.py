import torch
import time
import sys
import json

from models import NNDT_Sum_NNTG, MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
from loss import NllLoss, MyMSELoss
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 5  # locs are sampled from R^d
m = 50
input_trans_type = "dist_direction_lastloc"
nfeatures = input_transformed_dim(d, input_trans_type)
use_NN_for_testing = False
use_NN_for_training = False 
n_replicates_for_training = 'all'  # can be 'all' or a positive integer 
cond_on_train = True 
loss_function = NllLoss()
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
    train_type = "simulation"  # ["simulation", "data"]
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
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_DT = [nfeatures, 64, 64, 16] 
    size_TG_krig_coeff = [size_DT[-1] * 2, 64, 64, 1]
    size_TG_cond_sd_inv = [size_DT[-1], 64, 64, 1]
    n_batch = 1024
    n_epoch = 4001
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
model_krig_coeff = NNDT_Sum_NNTG(size_DT, size_TG_krig_coeff, concat_input=True)
model_cond_sd_inv = NNDT_Sum_NNTG(size_DT, size_TG_cond_sd_inv, concat_input=False)
model_krig_coeff.to(device)
model_cond_sd_inv.to(device)
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001
# %% model training
optimizer = torch.optim.Adam(
    list(model_krig_coeff.parameters()) + list(model_cond_sd_inv.parameters()), lr=1
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_krig_coeff.train()
model_cond_sd_inv.train()
timer = time.perf_counter()
for epoch in range(n_epoch):
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(
            size=n_batch, m=m, n_replicates=n_replicates_for_training, use_NN=use_NN_for_training
            )
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
model_krig_coeff.to("cpu")
model_cond_sd_inv.to("cpu")
loss_MSE = torch.nn.MSELoss()
loss_NLL = NllLoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(
            size=n_batch*10, m=m
            )
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
            "d": d,
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "NN",
            "m": m,
            "size_DT": size_DT,
            "size_TG_krig_coeff": size_TG_krig_coeff,
            "size_TG_cond_sd_inv": size_TG_cond_sd_inv,
            "transformation": input_trans_type,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "model": "NN",
            "m": m,
            "size_DT": size_DT,
            "size_TG_krig_coeff": size_TG_krig_coeff,
            "size_TG_cond_sd_inv": size_TG_cond_sd_inv,
            "transformation": input_trans_type,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")

# %%
