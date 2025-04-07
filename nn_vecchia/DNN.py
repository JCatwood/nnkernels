import torch
import gpytorch

torch.manual_seed(1)

def prepare_sequence(length, kernel, nbatch = 1):
    locs = torch.rand([nbatch, length, 2])
    covmat = kernel(locs)
    L = torch.linalg.cholesky(covmat)
    return locs, L.to_dense()[:, -1, -1]

class DNNKernel(torch.nn.Module):
    def __init__(self, nFeature, nHidden):
        super(DNNKernel, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = nFeature
        self.feature2hidden = torch.nn.Sequential(
            torch.nn.Linear(self.n_feature, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.Tanh()
        )
        self.hidden2pred = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, 1)
        )

    def forward(self, X):
        hidden_all = self.feature2hidden(X)
        hidden_sum = torch.sum(hidden_all, dim=-2)
        pred = self.hidden2pred(hidden_sum)
        cond_sd = torch.exp(pred).reshape((-1,))
        return cond_sd
    
model = DNNKernel(2, 128)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epoch = 10000
n_batch = 1000
kernel = gpytorch.kernels.MaternKernel(1.5)

model.train()
for epoch in range(n_epoch):  
    n_locs = 10
    # n_locs = random.randint(1, 30)
    locs, cond_sd = prepare_sequence(n_locs, kernel, n_batch)
    optimizer.zero_grad()
    cond_sd_pred = model(locs)
    loss = loss_function(cond_sd_pred, cond_sd)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Loss after {epoch} iterations is {loss.detach().item()}", flush=True)

model.eval()
with torch.no_grad():
    # for n_locs in range(1, 31):
    for n_locs in range(10, 11):
        locs, cond_sd = prepare_sequence(n_locs, kernel, n_batch)
        cond_sd_pred = model(locs)
        loss = loss_function(cond_sd_pred, cond_sd)
        print(f"At length {n_locs}, var is {cond_sd.var().item()}, MSE is {loss.item()}")
