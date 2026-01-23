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
if len(sys.argv) > 4:
    if sys.argv[1].lower().strip() in ("yes", "true", "t", "y", "1", "on"):
        fixed_len = True
    elif sys.argv[1].lower().strip() in ("no", "false", "f", "n", "0", "off"):
        fixed_len = False
    else:
        raise ValueError(f"Invalid boolean value: '{sys.argv[1]}'")
    kernel_train_name = sys.argv[2]
    assert kernel_train_name in (
        "MyMaternKernel",
        "MyNSKernel_Scale",
        "MyNSKernel_Lengthscale",
    ), "Invalid kernel_train_name (second) arguments"
    train_type = sys.argv[3]
    assert train_type in ("data", "simulation"), "Invalid train_type (third) argument"
    if train_type == "simulation":
        kernel_gen_name = sys.argv[4]
        assert kernel_gen_name in (
            "MyMaternKernel",
            "MyNSKernel_Scale",
            "MyNSKernel_Lengthscale",
        ), "Invalid kernel_gen_name (fourth) arguments"
    else:
        data_name = sys.argv[4]
        if len(sys.argv) > 5:
            data_seed = int(sys.argv[5])
        else:
            data_seed = None
else:
    fixed_len = True
    train_type = "simulation"  # ["simulation", "data"]
    data_name = "GP_NS_scale_2000_1000"
    data_seed = 0
    kernel_gen_name = "MyNSKernel_Scale"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
    kernel_train_name = "MyMaternKernel"  # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]

# %% model parameters
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    n_batch = 2048
    n_epoch = 30000
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
    n_batch = 1024
    n_epoch = 4000
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
    if data_seed is None:
        dataloader = Vecc_Dataloader_Dataset(data_name, fixed_len, length_max=m + 1)
    else:
        dataloader = Vecc_Dataloader_Dataset(
            data_name, fixed_len, length_max=m + 1, seed=data_seed
        )
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
        X_batch, y_batch, y_true, length = dataloader.get_minibatch(size=n_batch)
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
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
# %% evaluate
model.to("cpu")
model.eval()
loss_MSE = torch.nn.MSELoss()
with torch.no_grad():
    if train_type == "simulation":
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size=n_batch)
    else:
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(
            size=dataloader.n_test
        )
    y_pred, y_stderr = model(X_batch, y_batch, length=length)
    loss = loss_function(y_pred, y_true, y_stderr)
    print(">>>")
    if train_type == "simulation":
        output_dict = {
            "data_type": train_type,
            "kernel_sim": kernel_gen_name,
            "model": "GPVecchia",
            "m": m,
            "same_length": fixed_len,
            "NLL": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    else:
        output_dict = {
            "data_type": train_type,
            "data_name": data_name,
            "seed": data_seed,
            "model": "GPVecchia",
            "m": m,
            "same_length": fixed_len,
            "NLL": loss.detach().item(),
            "MSE": loss_MSE(y_pred, y_true).item(),
        }
    output_str = json.dumps(output_dict)
    print(output_str)
    print("<<<")
