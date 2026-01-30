import torch
import sys
import time
import json

from dataloader import Vecc_Dataloader_GP_sim
from models import NNDT2_Sum_NNTG, MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
d = 2 # locs are sampled from R^d
target = 'inv_chol'
fixed_len = False
input_trans_type = 'dist_direction_lastloc'
aggregate_mtd = 'sum'
m_max = 50
nfeatures = input_transformed_dim(d, input_trans_type)
kernel_gen_name = "MyMaternKernel" # ["MyMaternKernel", "MyNSKernel_Scale", "MyNSKernel_Lengthscale"]
# %% model parameters
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_DT1 = [nfeatures, 128, 128, 128, 16] 
    size_DT2 = [nfeatures, 128, 128, 128, 16] 
    size_TG = [size_DT1[-1] + size_DT2[-1], 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 40001
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_DT1 = [nfeatures, 64, 64, 8] 
    size_DT2 = [nfeatures, 64, 64, 8] 
    size_TG = [size_DT1[-1] + size_DT2[-1], 64, 64, 1]
    n_batch = 1024
    n_epoch = 5001
# %% define covariance kernel
# %% dataloader
if kernel_gen_name == "MyMaternKernel":
    kernel_parms_init = [1.0, 0.3, 1.5, 0.01]
    KernelClass = MyMaternKernel
elif kernel_gen_name == "MyNSKernel_Scale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 0.3, 1.5, 0.01]
    KernelClass = MyNSKernel_Scale
elif kernel_gen_name == "MyNSKernel_Lengthscale":
    kernel_parms_init = [-0.5, -1.2, -1.44, 2.0, 0.01]
    KernelClass = MyNSKernel_Lengthscale
dataloader = Vecc_Dataloader_GP_sim(KernelClass, kernel_parms_init, d, 
                                    fixed_length=True, length_max=m_max + 1, target=target)
# %% initialize model
model = NNDT2_Sum_NNTG(size_DT1, size_DT2, size_TG)
model.to(device)
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr/(1+factor*epoch)
    # return 0.001
# %% loss func
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# %% model training
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    with torch.no_grad():
        if fixed_len:
            locs_batch, y_batch, y, length = dataloader.get_minibatch(size=n_batch)
        else:
            len_iter = torch.randint(2, m_max + 2, (1, )).item()
            locs_batch, y_batch, y, length = dataloader.get_minibatch(size=n_batch, length_max=len_iter)
        X_batch_trans = input_transform(locs_batch, None, length, input_trans_type)
        X_batch_trans = X_batch_trans[:, :-1, :]
        y = - y[:, :-1, :] / y[:, -1:, :]
        X_batch_trans, y = X_batch_trans.to(device), y.to(device)
    # predict the target
    optimizer.zero_grad()
    y_pred = model(X_batch_trans)    
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        print(f"Total variation of y is {y.var().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
        crt_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {crt_lr}", flush=True)

model.eval()
model.to('cpu')
if not fixed_len:
    with torch.no_grad():
        losses = []
        for len_iter in range(2, m_max + 2):
            locs_batch, y_batch, y, length = dataloader.get_minibatch(size=n_batch, length_max=len_iter)
            X_batch_trans = input_transform(locs_batch, None, length, input_trans_type)
            X_batch_trans = X_batch_trans[:, :-1, :]
            y = - y[:, :-1, :] / y[:, -1:, :]
            y_pred = model(X_batch_trans)    
            loss = loss_function(y_pred, y)
            losses.append(loss)
    loss = torch.mean(torch.tensor(losses))
model_fn = f"NN2_krig_coeff_d{d}_{input_trans_type}_{kernel_gen_name}_loss{loss.detach().item():.4f}"
model_state_fn = model_fn + ".pt"
model_init_parm_fn = model_fn + ".json"
with open(model_init_parm_fn, 'w') as json_file:
    json.dump({
        "size_DT1": size_DT1,
        "size_DT2": size_DT2,
        "size_TG": size_TG,
    }, json_file)
torch.save(model.state_dict(), model_state_fn)