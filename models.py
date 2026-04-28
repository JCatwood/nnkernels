import torch 
import DeepKernelNNGP

def init_DeepKernelNNGP(d, dropout=0.0, device=torch.device("cpu"),
                        size=None):
    MeanClass = DeepKernelNNGP.ConstMean
    CovClass = DeepKernelNNGP.NNKernel
    input_trans_type = 'locs_lastloc'
    nfeatures = DeepKernelNNGP.input_transformed_dim(d, input_trans_type)
    size_configs = {
        "small": {"latent_dim": 32, "dim_middle": 64},
        "big": {"latent_dim": 64, "dim_middle": 128},
    }

    if size is None:
        size = "big" if device.type == "cuda" else "small"

    if size not in size_configs:
        raise ValueError(
            f"Invalid size={size!r}. Expected one of {list(size_configs)} or None."
        )

    latent_dim = size_configs[size]["latent_dim"]
    dim_middle = size_configs[size]["dim_middle"]
    nlayer_middle_rho = 3
    nlayer_middle_phi = 3
    nlayer_middle_rho1 = 3
    nlayer_middle_rho2 = 3
    mean_class_init = [0.0]
    cov_class_init = [nfeatures, dim_middle, latent_dim, nlayer_middle_rho, 
                    nlayer_middle_phi, nlayer_middle_rho1, nlayer_middle_rho2, dropout]
    model_specs = {
        'MeanClass': MeanClass.__name__,
        'mean_class_init': mean_class_init,
        'CovClass': CovClass.__name__,
        'cov_class_init': cov_class_init,
        'input_trans_type': input_trans_type
    }
    model = DeepKernelNNGP.DeepKernelNNGP(MeanClass, CovClass, mean_class_init, cov_class_init, input_trans_type)
    return model, model_specs

def init_VGP(d, device=torch.device("cpu")):
    MeanClass = DeepKernelNNGP.ConstMean
    CovClass = DeepKernelNNGP.MyMaternKernel
    mean_class_init = [0.0]
    cov_class_init = [0.5, [0.1 for i in range(d)], 1.5, 0.01]
    model_specs = {
        'MeanClass': MeanClass.__name__,
        'mean_class_init': mean_class_init,
        'CovClass': CovClass.__name__,
        'cov_class_init': cov_class_init,
    }
    model = DeepKernelNNGP.GPVecchia(MeanClass, CovClass, mean_class_init, cov_class_init)
    return model, model_specs

def init_VGP_SM(d, device=torch.device("cpu")):
    MeanClass = DeepKernelNNGP.ConstMean
    CovClass = DeepKernelNNGP.SpectralMixtureKernel
    mean_class_init = [0.0]
    cov_class_init = [[0.1] * 6] + [[[0 for _ in range(d)] for _ in range(6)]] + \
        [[[1 for _ in range(d)] for _ in range(6)]] + [0.01]
    model_specs = {
        'MeanClass': MeanClass.__name__,
        'mean_class_init': mean_class_init,
        'CovClass': CovClass.__name__,
        'cov_class_init': cov_class_init,
    }
    model = DeepKernelNNGP.GPVecchia(MeanClass, CovClass, mean_class_init, cov_class_init)
    return model, model_specs

def init_VGP_Wilson2015Deep(d, device=torch.device("cpu")):
    MeanClass = DeepKernelNNGP.ConstMean
    CovClass = DeepKernelNNGP.Wilson2015Deep
    mean_class_init = [0.0]
    cov_class_init = [[d, 1000, 1000, 500, 50, 2]] + [[0.1] * 6] + \
        [[[0 for _ in range(2)] for _ in range(6)]] + \
        [[[1 for _ in range(2)] for _ in range(6)]] + [0.01]
    model_specs = {
        'MeanClass': MeanClass.__name__,
        'mean_class_init': mean_class_init,
        'CovClass': CovClass.__name__,
        'cov_class_init': cov_class_init,
    }
    model = DeepKernelNNGP.GPVecchia(MeanClass, CovClass, mean_class_init, cov_class_init)
    return model, model_specs