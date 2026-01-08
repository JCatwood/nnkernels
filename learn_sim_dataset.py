import torch
import gpytorch
import time

from models import NNDT_Sum_NNTG, GPVecchia, MyMaternKernel
from loss import NllLoss
from data_sim import prepare_sequence_locs_and_y

torch.manual_seed(1)

d = 2 # locs are sampled from R^d
m = 30
dropout_ratio = 0.0
fixed_len = True

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    size_DT_mean = [d + 1, 128, 128, 128, 8]
    size_TG_mean = [8, 128, 128, 128, 1]
    size_DT_sd = [d, 128, 128, 128, 8]
    size_TG_sd = [8, 128, 128, 128, 1]
    n_batch = 2048
    n_epoch = 30000
else:
    print("GPU is not available. Using CPU.")
    device = torch.device('cpu')
    size_DT_mean = [d + 1, 96, 96, 4] 
    size_TG_mean = [4, 96, 96, 1]
    size_DT_sd = [d, 64, 64, 3] 
    size_TG_sd = [3, 32, 32, 1]
    n_batch = 1024
    n_epoch = 4000

kernel = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
model_mean = NNDT_Sum_NNTG(size_DT_mean, size_TG_mean, dropout=dropout_ratio)
model_sd = NNDT_Sum_NNTG(size_DT_sd, size_TG_sd, dropout=dropout_ratio)
model_mean.to(device)
model_sd.to(device)

# scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr/(1+factor*epoch)
    # return 0.001

loss_function = NllLoss()
optimizer = torch.optim.Adam([{'params': model_mean.parameters()}, 
                              {'params': model_sd.parameters()}], lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_mean.train()
model_sd.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    if fixed_len:
        length = m + 1
        length_max = m + 1
    else:
        length = torch.randint(1, m + 1, (n_batch,))
        length_max = length.max().item()
    # draw mini-batches
    with torch.no_grad():
        locs_batch, y_batch = prepare_sequence_locs_and_y(length_max, kernel, nbatch=n_batch, d=d)
        locs_at_length = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        locs_batch = locs_batch - locs_at_length
        y_true = y_batch[torch.arange(n_batch), length - 1].clone()
        y_batch[torch.arange(n_batch), length - 1] = 0.0
        locs_and_y_batch = torch.cat((locs_batch, y_batch.unsqueeze(-1)), dim=-1)
        locs_batch, y_true, locs_and_y_batch = locs_batch.to(device), \
            y_true.to(device), locs_and_y_batch.to(device)
    # predict mean and stderr
    optimizer.zero_grad()
    y_pred = model_mean(locs_and_y_batch, length=length)
    y_stderr = torch.exp(model_sd(locs_batch, length=length))
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

model_mean.eval()
model_sd.eval()
# compare with GPVecchia initiated with the true kernel
with torch.no_grad():
    if fixed_len:
        length = m + 1
        length_max = m + 1
    else:
        length = torch.randint(1, m + 1, (n_batch,))
        length_max = length.max().item()
    locs_batch, y_batch = prepare_sequence_locs_and_y(length_max, kernel, nbatch=n_batch, d=d)
    locs_at_length = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
    locs_batch = locs_batch - locs_at_length
    y_true = y_batch[torch.arange(n_batch), length - 1].clone()
    y_batch[torch.arange(n_batch), length - 1] = 0.0
    locs_and_y_batch = torch.cat((locs_batch, y_batch.unsqueeze(-1)), dim=-1)
    locs_batch, y_true, locs_and_y_batch, y_batch = locs_batch.to(device), \
        y_true.to(device), locs_and_y_batch.to(device), y_batch.to(device)
    
    y_pred = model_mean(locs_and_y_batch, length=length)
    y_stderr = torch.exp(model_sd(locs_batch, length=length))
    loss = loss_function(y_pred, y_true, y_stderr)
    print(f"Loss of the proposed model is {loss.detach().item()}", flush=True)
    
    mdl_GP = GPVecchia(MyMaternKernel, *[1.0, 0.3, 1.5, 0.01]).to(device)
    y_pred_test_GP, y_stderr_test_GP = mdl_GP(locs_batch, y_batch, length=length)
    loss_test_GP = loss_function(y_pred_test_GP, y_true, y_stderr_test_GP)
    print(f"Loss of GP using the testing dataset is {loss_test_GP.detach().item()}", flush=True)

    mdl_GP = GPVecchia(MyMaternKernel, *[1.0, 0.1, 0.5, 0.005]).to(device)
    y_pred_test_GP, y_stderr_test_GP = mdl_GP(locs_batch, y_batch, length=length)
    loss_test_GP = loss_function(y_pred_test_GP, y_true, y_stderr_test_GP)
    print(f"Loss of mis-specified GP using the testing dataset is {loss_test_GP.detach().item()}", flush=True)