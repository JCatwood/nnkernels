import torch
import gpytorch
import sys
import time

from data_sim import prepare_sequence_cond_sd, prepare_sequence_cond_mean
from models import NNDT_Sum_NNTG, MyMaternKernel

torch.manual_seed(1)

d = 2 # locs are sampled from R^d
target = 'cond_sd'
fixed_len = False
if len(sys.argv) > 1:
    n_layer_DT = int(sys.argv[1])
    n_layer_TG = int(sys.argv[2])
    size_all = [int(sys.argv[i]) for i in range(3, len(sys.argv))]
    assert len(size_all) == n_layer_DT + n_layer_TG
    assert size_all[n_layer_DT - 1] == size_all[n_layer_DT]
    if target == 'cond_sd':
        size_ST = [d] + size_all[:n_layer_DT]
    else: # cond_mean
        size_ST = [d + 1] + size_all[:n_layer_DT]
    size_TG = size_all[n_layer_DT:] + [1]
else:
    n_layer_DT = 3
    n_layer_TG = 3
    # the latent dim is 3
    if target == 'cond_sd':
        size_ST = [d, 64, 64, 3] 
        size_TG = [3, 32, 32, 1]
    else:
    # the latent dim is 4
        size_ST = [d + 1, 96, 96, 4] 
        size_TG = [4, 96, 96, 1]
n_epoch = 4000
n_batch = 1000

if target == "cond_sd":
    seq_func = prepare_sequence_cond_sd
else:
    seq_func = prepare_sequence_cond_mean
model = NNDT_Sum_NNTG(size_ST, size_TG)

# scheduler
def lr_lambda(epoch):
    # base_lr = 0.001
    # factor = 0.001
    # return base_lr/(1+factor*epoch)
    return 0.001

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
kernel_gpt = gpytorch.kernels.MaternKernel(1.5)
kernel_gpt.lengthscale = 0.3
kernel = MyMaternKernel(1.0, 0.3, 1.5, 0.01)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
model.to(device)

model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    # simulate training data
    if fixed_len:
        length = 10
    else:
        length = torch.randint(1, 30, (n_batch,))
    with torch.no_grad():
        X, y = seq_func(length, kernel, nbatch=n_batch, d=d)
        if target == 'cond_mean':
            X[torch.arange(n_batch), length - 1, d] = 0.0
        X_at_length = X[torch.arange(n_batch), length - 1, :] # n_batch X d or n_batch X (d + 1)
        X_at_length_reshape = X_at_length.reshape(n_batch, 1, -1)
        X = X - X_at_length_reshape
        X, y = X.to(device), y.to(device)
    # predict the target
    optimizer.zero_grad()
    if target == 'cond_sd':
        y_pred = torch.exp(model(X, length))
    else:
        y_pred = model(X, length)
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
print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
print(f"Total variation of y is {y.var().item()}", flush=True)
timer_prev = timer
timer = time.perf_counter()
print(f"Elapsed time: {timer - timer_prev} seconds", flush=True)
crt_lr = optimizer.param_groups[0]["lr"]
print(f"Current LR: {crt_lr}", flush=True)