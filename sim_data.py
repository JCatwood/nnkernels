import torch
import pandas
import os
from models import MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
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
        locs = torch.rand(n, d)
    mean_y = mean_obj(locs)
    covmat = kernel(locs).to_dense()
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
    n_train = 2000
    n_test = 1000
    d = 2
    mean_obj_mean0 = ZeroMean()
    mean_obj_Paraboloid = ParaboloidMean()
    kernel_Matern = MyMaternKernel(1.0, 0.3, 1.5, 0.01)
    kernel_NS_scale = MyNSKernel_Scale(-0.5, -1.2, -1.44, 0.3, 1.5, 0.01)
    kernel_NS_lengthrange = MyNSKernel_Lengthscale(-0.5, -1.2, -1.44, 2.0, 0.01)
    kernel_and_name = zip([kernel_Matern, kernel_NS_scale, kernel_NS_lengthrange], 
                          ["Matern", "NS_scale", "NS_range"])
    seeds = range(20)
    N = len(seeds)
    offset_train = torch.arange(N) * n_train
    offset_test = torch.arange(N) * n_test
    for kernel, name in kernel_and_name:
        for seed in seeds:
            sim_GP_data(mean_obj_mean0, kernel, n_train, n_test, d=d,
                        mean_name="mean0", kernel_name=name, seed=seed, mode='a')
        fn = file_name(n_train, n_test, d, "mean0", name, locs=None)
        df_offset_train = pandas.DataFrame(offset_train)
        df_offset_test = pandas.DataFrame(offset_test)
        df_offset_train.to_csv(fn + "train_offset.csv", index=False, header=False)
        df_offset_test.to_csv(fn + "test_offset.csv", index=False, header=False)
    
    kernel_and_name = zip([kernel_Matern, kernel_NS_scale, kernel_NS_lengthrange], 
                          ["Matern", "NS_scale", "NS_range"])
    seeds_train = range(16)
    seeds_test = range(16, 20)
    n = 2000
    locs = torch.from_numpy(LatinHypercube(d).random(n)).float()
    offset_train = torch.arange(len(seeds_train)) * n
    offset_test = torch.arange(len(seeds_test)) * n
    for kernel, name in kernel_and_name:
        fn = file_name(len(seeds_train), len(seeds_test), d, "mean0", name, locs=locs)
        for seed in seeds_train:
            sim_GP_data(mean_obj_mean0, kernel, n_train=n, n_test=0, d=d,
                        mean_name="mean0", kernel_name=name, seed=seed, mode='a',
                        locs=locs, fn=fn)
        for seed in seeds_test:
            sim_GP_data(mean_obj_mean0, kernel, n_train=0, n_test=n, d=d,
                        mean_name="mean0", kernel_name=name, seed=seed, mode='a',
                        locs=locs, fn=fn)
        df_offset_train = pandas.DataFrame(offset_train)
        df_offset_test = pandas.DataFrame(offset_test)
        df_offset_train.to_csv(fn + "train_offset.csv", index=False, header=False)
        df_offset_test.to_csv(fn + "test_offset.csv", index=False, header=False)
