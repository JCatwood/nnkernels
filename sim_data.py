import torch
import pandas
import os
from models import MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale

class ZeroMean(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        if x.dim() == 3:
            B, n, d = x.size(0), x.size(1), x.size(2)
            return torch.zeros(B, n, 1)
        else:
            n, d = x.size(0), x.size(1)
            return torch.zeros(n, 1)


class ParaboloidMean(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.sum(x ** 2, dim=-1, keepdim=True)


def sim_GP_data(mean_obj, kernel, n_train, n_test, d=2, mean_name="meanname", kernel_name="kernelname", seed=1):
    """
    Simulate spatial GP data where locs are randomly sampled from the unit hypercube in R^d
    """
    torch.manual_seed(seed)

    fn = f"./data/GP_d{d}_{mean_name}_{kernel_name}_{n_train}_{n_test}/seed_{seed}/"
    os.makedirs(fn + "train/", exist_ok=True)
    os.makedirs(fn + "test/", exist_ok=True)

    n = n_train + n_test
    locs = torch.rand(n, d)
    mean_y = mean_obj(locs)
    covmat = kernel(locs).to_dense()
    L = torch.linalg.cholesky(covmat)
    x = torch.normal(0.0, 1.0, (n, 1))
    y = L @ x + mean_y

    df_x_train = pandas.DataFrame(locs[:n_train, :].detach().numpy())
    df_y_train = pandas.DataFrame(y[:n_train, :].detach().numpy())
    df_x_test = pandas.DataFrame(locs[n_train:, :].detach().numpy())
    df_y_test = pandas.DataFrame(y[n_train:, :].detach().numpy())

    df_x_train.to_csv(fn + "train/x.csv", index=False, header=False)
    df_y_train.to_csv(fn + "train/y.csv", index=False, header=False)
    df_x_test.to_csv(fn + "test/x.csv", index=False, header=False)
    df_y_test.to_csv(fn + "test/y.csv", index=False, header=False)

if __name__ == "__main__":
    # GP
    n_train = 2000
    n_test = 1000
    d = 3
    mean_obj_mean0 = ZeroMean()
    mean_obj_Paraboloid = ParaboloidMean()
    kernel_Matern = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
    kernel_NS_scale = MyNSKernel_Scale(-0.5, -1.2, -1.44, 0.3, 1.5, 0.01)
    kernel_NS_lengthrange = MyNSKernel_Lengthscale(-0.5, -1.2, -1.44, 2.0, 0.01)
    kernel_and_name = zip([kernel_Matern, kernel_NS_scale, kernel_NS_lengthrange], 
                          ["Matern", "NS_scale", "NS_range"])
    for kernel, name in kernel_and_name:
        for seed in torch.arange(20):
            sim_GP_data(mean_obj_mean0, kernel, n_train, n_test, d=d,
                        mean_name="mean0", kernel_name=name, seed=seed)
            sim_GP_data(mean_obj_Paraboloid, kernel, n_train, n_test, d=d,
                        mean_name="Paraboloid", kernel_name=name, seed=seed)
