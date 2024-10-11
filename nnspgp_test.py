import torch
import gpytorch
import pandas
import argparse
from torch import nn

torch.manual_seed(123)

class NNSPGP(nn.Module):
    def __init__(self, d: int, m: int, middle_width: int = 32):
        super().__init__()
        self.d = d
        self.m = m
        self.middle_width = middle_width
        self.nugget = nn.Parameter(torch.Tensor([0.1]))
        self.linear_stack = nn.Sequential(
            nn.Linear(d, self.middle_width),
            nn.ReLU(),
            nn.Linear(self.middle_width, self.middle_width),
            nn.ReLU(),
            nn.Linear(self.middle_width, m),
        )

    def forward(self, x):
        lin_coeff = self.linear_stack(x)
        covmat_nn = lin_coeff @ lin_coeff.T + \
            torch.eye(x.size(0)) * self.nugget
        return gpytorch.distributions.MultivariateNormal(torch.zeros([x.size(0)]), covmat_nn)
    
    def update(self, m):
        if m < self.m:
            print("update only applies to input m bigger than self.m")
        else:
            m_prev = self.m
            self.m = m
            tmp = self.linear_stack[-1]
            self.linear_stack[-1] = nn.Linear(self.middle_width, m)
            # self.linear_stack[-1].weight[:m_prev, ] = tmp.weight
            # self.linear_stack[-1].bias[:m_prev] = tmp.bias
            # self.linear_stack[-1].weight[m_prev:, ] = 0
            # self.linear_stack[-1].bias[m_prev:] = 0

with torch.no_grad():
    m_vec = torch.arange(3, 31, step=3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="1Dexponential_heteroskedasticity")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--d", type=int, default=1)
    args = parser.parse_args()
    scene = args.scene
    N = args.N
    d = args.d
    data = torch.tensor(pandas.read_csv(
        f"data/{scene}/0/train.csv", header=None, index_col=None).values, 
        dtype=torch.float64)
    locs_train = data[:, :d]
    data = torch.tensor(pandas.read_csv(
        f"data/{scene}/0/test.csv", header=None, index_col=None).values,
        dtype=torch.float64)
    locs_test = data[:, :d]
    y_train = torch.zeros([N, locs_train.size(0)], dtype=torch.float64)
    y_test = torch.zeros([N, locs_test.size(0)], dtype=torch.float64)
    for k in range(N):
        data = torch.tensor(pandas.read_csv(
            f"data/{scene}/{k}/train.csv", header=None, index_col=None).values,
            dtype=torch.float64)
        y_train[k, :] = data[:, d]
        data = torch.tensor(pandas.read_csv(
            f"data/{scene}/{k}/test.csv", header=None, index_col=None).values,
            dtype=torch.float64)
        y_test[k, :] = data[:, d]

for m in m_vec:
    with torch.no_grad():
        model = NNSPGP(d, m)
        model.double()
    model.train()
    optimizer = torch.optim.LBFGS([
        {'params': model.parameters()},
    ], lr = 0.01, line_search_fn='strong_wolfe', tolerance_change=1e-4)
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        mvn_dist = model(locs_train)
        loss = - mvn_dist.log_prob(y_train).mean()
        if loss.requires_grad:
            loss.backward()
        return loss
    for i in range(500):
        optimizer.step(closure)

    model.eval()
    with torch.no_grad():
        # loss = closure()
        # print(f"m = {m}, llk is {-loss}")

        lin_coeff_train = model.linear_stack(locs_train)
        lin_coeff_test = model.linear_stack(locs_test)
        covmat_test_train = lin_coeff_test @ lin_coeff_train.T
        covmat_train = lin_coeff_train @ lin_coeff_train.T + \
            torch.eye(locs_train.size(0)) * model.nugget
        y_pred = (covmat_test_train @ torch.linalg.solve(covmat_train, y_train.T)).T
        mse = torch.mean(pow(y_test - y_pred, 2))
        print(f"{m},{mse}")






