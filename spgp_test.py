import torch
import gpytorch
import pandas
import argparse

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.manual_seed(123)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

with torch.no_grad():
    m_vec = torch.arange(3, 31, step=3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="1Dexponential_heteroskedasticity")
    parser.add_argument("--N", type=int, default=2)
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
    mse = torch.zeros([N])
    for k in range(N):
        inducing_points = locs_train[:m, ]
        model = GPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.double()
        likelihood.double()
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=locs_train.size(0))
        for i in range(1000):
            optimizer.zero_grad()
            output = model(locs_train)
            loss = -mll(output, y_train[k, :])
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = model(locs_test)
            means = preds.mean
            mse[k] = torch.mean(pow(y_test[k, :] - means, 2))
    with torch.no_grad():
        print(f"{m},{mse.mean()}")