import torch
import gpytorch
import os
import sys
import time

from loss_func import prepare_sequence_cond_sd, prepare_sequence_cond_mean
from loss_func import loss_len10_cond_sd, loss_len1to30_cond_sd
from loss_func import loss_len10_cond_mean, loss_len1to30_cond_mean

torch.manual_seed(1)

class LSTMKernelSD(torch.nn.Module):
    def __init__(self, locsDim, nHidden):
        super(LSTMKernelSD, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = locsDim
        self.lstm = torch.nn.LSTM(locsDim, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, locs, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(locs)
        else:
            lstm_out, (h_out, c_out) = self.lstm(locs, (h0, c0))
        out = torch.exp(self.hidden2pred(lstm_out[:, -1, :])).reshape((-1,))
        return out, h_out, c_out

class LSTMKernelMean1(torch.nn.Module):
    def __init__(self, locsNyDim, nHidden):
        super(LSTMKernelMean1, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = locsNyDim
        self.lstm = torch.nn.LSTM(locsNyDim, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, locsNy, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(locsNy)
        else:
            lstm_out, (h_out, c_out) = self.lstm(locsNy, (h0, c0))
        out = self.hidden2pred(lstm_out[:, -1, :]).reshape((-1,))
        return out, h_out, c_out
    
class LSTMKernelMean2(torch.nn.Module):
    def __init__(self, locsNyDim, nHidden):
        super(LSTMKernelMean2, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = locsNyDim - 1
        self.lstm = torch.nn.LSTM(locsNyDim - 1, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, locsNy, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(locsNy[:, :, :-1])
        else:
            lstm_out, (h_out, c_out) = self.lstm(locsNy[:, :, :-1], (h0, c0))
        coeff = self.hidden2pred(lstm_out)
        out = torch.sum(coeff * locsNy[:, :, -1:], dim=1).squeeze()
        return out, h_out, c_out

if len(sys.argv) > 1:
    n_hidden = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    n_batch = int(sys.argv[3])
else:
    n_hidden = 256
    n_epoch = 4000
    n_batch = 100
fixed_len = False
task_type = "cond_mean" 
if task_type == "cond_sd":
    model = LSTMKernelSD(2, n_hidden)
    seq_func = prepare_sequence_cond_sd
    if fixed_len:
        loss_func = loss_len10_cond_sd
        output_fn = f"trained_models/cond_sd_len10_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
    else:
        loss_func = loss_len1to30_cond_sd
        output_fn = f"trained_models/cond_sd_len1-30_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
else: # "cond_mean"
    model = LSTMKernelMean1(3, n_hidden)
    seq_func = prepare_sequence_cond_mean
    if fixed_len:
        loss_func = loss_len10_cond_mean
        output_fn = f"trained_models/cond_mean_len10_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"
    else:
        loss_func = loss_len1to30_cond_mean
        output_fn = f"trained_models/cond_mean_len1-30_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt"

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kernel = gpytorch.kernels.MaternKernel(1.5)

os.makedirs("trained_models", exist_ok=True)
model.train()
timer = time.perf_counter()
for epoch in range(n_epoch): 
    optimizer.zero_grad()
    loss = loss_func(model, loss_function, kernel, n_batch)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)
        timer_prev = timer
        timer = time.perf_counter()
        print(f"Elapsed time: {timer - timer_prev } seconds", flush=True)
torch.save(model.state_dict(), output_fn)

# model.eval()
# with torch.no_grad():
#     if fixed_len:
#         for n_locs in range(10, 11):
#             locs, cond_sd = prepare_sequence_cond_sd(n_locs, kernel, n_batch)
#             cond_sd_pred, _, _ = model(locs, None, None)
#             loss = loss_function(cond_sd_pred, cond_sd)
#             print(f"At length {n_locs}, var is {cond_sd.var().item()}, MSE is {loss.item()}")
#     else:
#         length = torch.randint(1, 30, (n_batch,))
#         locs, cond_sd = prepare_sequence_cond_sd(length, kernel, n_batch)
#         cond_sd_pred, _, _ = model(locs, None, None)
#         loss = loss_function(cond_sd_pred, cond_sd)
#         print(f"with n_loc in [{min(length)}, {max(length)}], "
#               f"var is {cond_sd.var().item()}, MSE is {loss.item()}")
