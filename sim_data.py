import torch
import pandas
import os
from DeepKernelNNGP import MyMaternKernel, MyNSKernel_Lengthscale, PeriodicKernel, TransformedMaternKernel
from scipy.stats.qmc import LatinHypercube

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


def file_name(n_train, n_test, d=2, mean_name="meanname", kernel_name="kernelname", locs=None):
    if locs is not None:
        fn = f"./data/GP_d{d}_fixedlocs_{mean_name}_{kernel_name}_{n_train}_{n_test}/"
    else:
        fn = f"./data/GP_d{d}_rndlocs_{mean_name}_{kernel_name}_{n_train}_{n_test}/"
    return fn


def sim_GP_data(mean_obj, kernel, n_train, n_test, d=2, mean_name="meanname", 
                kernel_name="kernelname", seed=0, locs=None, mode='w', fn=None):
    """
    Simulate spatial GP data where locs are randomly sampled from the unit hypercube in R^d
    """
    if fn is None:
        fn = file_name(n_train, n_test, d, mean_name, kernel_name, locs)
    n = n_train + n_test
    os.makedirs(fn + "train/", exist_ok=True)
    os.makedirs(fn + "test/", exist_ok=True)

    torch.manual_seed(seed)
    if locs is None:
        locs = torch.from_numpy(LatinHypercube(d).random(n)).float()
        locs_scaled = locs * ((n / 100)**(1/d))
    else:
        locs_scaled = locs
    mean_y = mean_obj(locs_scaled)
    covmat = kernel(locs_scaled).to_dense()
    L = torch.linalg.cholesky(covmat)
    x = torch.normal(0.0, 1.0, (n, 1))
    y = L @ x + mean_y

    df_x_train = pandas.DataFrame(locs[:n_train].detach().numpy())
    df_y_train = pandas.DataFrame(y[:n_train].detach().numpy())
    df_x_test = pandas.DataFrame(locs[n_train:].detach().numpy())
    df_y_test = pandas.DataFrame(y[n_train:].detach().numpy())

    df_x_train.to_csv(fn + "train/x.csv", index=False, header=False, mode=mode)
    df_y_train.to_csv(fn + "train/y.csv", index=False, header=False, mode=mode)
    df_x_test.to_csv(fn + "test/x.csv", index=False, header=False, mode=mode)
    df_y_test.to_csv(fn + "test/y.csv", index=False, header=False, mode=mode)

if __name__ == "__main__":
    # GP
    n = 2500
    n_train = 2000
    n_test = 500
    d = 3
    mean_obj_mean0 = ZeroMean()
    kernel_Matern = MyMaternKernel(1.0, [0.03] * d, 1.5, 0.01)
    kernel_NS_lengthrange = MyNSKernel_Lengthscale(-4.0, 4.0, -4.0, 1.0, 0.01)
    kernel_Periodic = PeriodicKernel(1.0, 0.5, d, 0.01)
    kernel_TransMatern = TransformedMaternKernel(d, 1.0, 0.1 * (d ** 0.5), 1.5, 0.01)
    kernel_and_name = zip([kernel_Matern, kernel_NS_lengthrange, kernel_Periodic, kernel_TransMatern], 
                          ["Matern", "NS_range", "Periodic", "TransMatern"])
    
    for kernel, kernel_name in kernel_and_name:
        fn_base = file_name(n_train, n_test, d, "mean0", kernel_name, locs=None)
        for seed in range(100):
            fn = fn_base + f"seed_{seed}/"
            sim_GP_data(mean_obj_mean0, kernel, n_train=n_train, n_test=n_test, d=d,
                        mean_name="mean0", kernel_name=kernel_name, seed=seed, mode='w',
                        locs=None, fn=fn)
