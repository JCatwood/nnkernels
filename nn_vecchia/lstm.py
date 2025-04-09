import torch
import gpytorch
import random
import os
import sys

torch.manual_seed(1)

def prepare_sequence(length, kernel, nbatch = 1):
    locs = torch.rand([nbatch, length, 2])
    covmat = kernel(locs)
    L = torch.linalg.cholesky(covmat)
    return locs, L.to_dense()[:, -1, -1]


class LSTMKernel(torch.nn.Module):
    def __init__(self, nFeature, nHidden):
        super(LSTMKernel, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = nFeature
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(nFeature, nHidden, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        # Do I need another activation/transformation layer?
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, X, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(X)
        else:
            lstm_out, (h_out, c_out) = self.lstm(X, (h0, c0))
        out = torch.exp(self.hidden2pred(lstm_out[:, -1, :])).reshape((-1,))
        return out, h_out, c_out

if len(sys.argv) > 1:
    n_hidden = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    n_batch = int(sys.argv[3])
else:
    n_hidden = 256
    n_epoch = 10000
    n_batch = 1000
model = LSTMKernel(2, n_hidden)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kernel = gpytorch.kernels.MaternKernel(1.5)

model.train()
for epoch in range(n_epoch):  
    n_locs = 10
    # n_locs = random.randint(1, 30)
    locs, cond_sd = prepare_sequence(n_locs, kernel, n_batch)
    optimizer.zero_grad()
    cond_sd_pred, _, _ = model(locs, None, None)
    loss = loss_function(cond_sd_pred, cond_sd)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)

os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), f"trained_models/cond_sd_len10_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt")
# torch.save(model.state_dict(), f"trained_models/cond_sd_len1-30_LSTM_{n_hidden}_{n_epoch}_{n_batch}.pt")

model.eval()
with torch.no_grad():
    # for n_locs in range(1, 31):
    for n_locs in range(10, 11):
        locs, cond_sd = prepare_sequence(n_locs, kernel, n_batch)
        cond_sd_pred, _, _ = model(locs, None, None)
        loss = loss_function(cond_sd_pred, cond_sd)
        print(f"At length {n_locs}, var is {cond_sd.var().item()}, MSE is {loss.item()}")
