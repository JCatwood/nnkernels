import torch
import gpytorch
import pandas
import scipy
import pathlib
from sklearn import model_selection

n = 200
d = 2
N = 100
train_ratio = 0.8

with torch.no_grad():
    locs_sampler = scipy.stats.qmc.LatinHypercube(d=d)
    locs = torch.tensor(locs_sampler.random(n))
    locs_sampler_1D = scipy.stats.qmc.LatinHypercube(d=1)
    locs_1D = torch.tensor(locs_sampler_1D.random(n))

    # scene = "matern15"
    # variance, lengthscale, smooth, nugget = 0.8, 0.2, 1.5, 0.2
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=smooth))
    # kernel.base_kernel._set_lengthscale(lengthscale)
    # kernel._set_outputscale(variance)
    # covmat = kernel(locs).evaluate() + torch.eye(n) * nugget
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)
    
    # scene = "exponential"
    # variance, lengthscale, smooth, nugget = 0.8, 0.2, 0.5, 0.2
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=smooth))
    # kernel.base_kernel._set_lengthscale(lengthscale)
    # kernel._set_outputscale(variance)
    # covmat = kernel(locs).evaluate() + torch.eye(n) * nugget
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)
    
    # scene = "PiecewisePolynomialKernel"
    # order = 2
    # kernel = gpytorch.kernels.PiecewisePolynomialKernel(order)
    # covmat = kernel(locs).evaluate() * 0.8 + torch.eye(n) * 0.2
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)
    
    # scene = "Constant"
    # constant_corr = torch.tensor([0.5])
    # kernel = gpytorch.kernels.ConstantKernel()
    # kernel.constant = constant_corr
    # covmat = kernel(locs).evaluate() + torch.eye(n) * 0.5
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)

    # scene = "RBFKernel"
    # kernel = gpytorch.kernels.RBFKernel()
    # kernel.lengthscale = 0.2
    # covmat = kernel(locs).evaluate() * 0.8 + torch.eye(n) * 0.2
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)
    
    # scene = "AdditiveKernel"
    # kernel = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([0])) + \
    #     gpytorch.kernels.MaternKernel(nu=0.5, active_dims=torch.tensor([1]))
    # covmat = kernel(locs).evaluate()
    # covmat = covmat / covmat[0, 0] * 0.8 + torch.eye(n) * 0.2 
    # chol = torch.linalg.cholesky(covmat, upper=False)
    # y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    # locs_train, locs_test, y_train, y_test = \
    #     model_selection.train_test_split(locs, y, train_size=train_ratio)
    # for k in range(N):
    #     pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
    #     data_train = pandas.DataFrame(
    #         torch.column_stack((locs_train, y_train[:, k])).numpy())
    #     data_test = pandas.DataFrame(
    #         torch.column_stack((locs_test, y_test[:, k])).numpy())
    #     data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
    #     data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)
    
    scene = "1Dexponential_heteroskedasticity"
    kernel = gpytorch.kernels.MaternKernel(nu=0.5)
    kernel.lengthscale = 0.1
    covmat = kernel(locs_1D).evaluate()
    diag_std = torch.diag(torch.sin(locs_1D[:, 0] * 4 * torch.pi) + 0.01)
    covmat = diag_std @ (covmat / covmat[0, 0] * 0.8 + torch.eye(n) * 0.2) @ diag_std
    chol = torch.linalg.cholesky(covmat, upper=False)
    y = chol @ torch.normal(0, 1, [n, N], dtype=chol.dtype)
    locs_train, locs_test, y_train, y_test = \
        model_selection.train_test_split(locs_1D, y, train_size=train_ratio)
    for k in range(N):
        pathlib.Path(f"./data/{scene}/{k}").mkdir(parents=True, exist_ok=True)
        data_train = pandas.DataFrame(
            torch.column_stack((locs_train, y_train[:, k])).numpy())
        data_test = pandas.DataFrame(
            torch.column_stack((locs_test, y_test[:, k])).numpy())
        data_train.to_csv(f"./data/{scene}/{k}/train.csv", index=False, header=False)
        data_test.to_csv(f"./data/{scene}/{k}/test.csv", index=False, header=False)


    