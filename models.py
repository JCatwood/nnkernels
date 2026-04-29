import torch 
import DeepKernelNNGP

def _get_mean_config(case, nfeatures, dropout=0.0):
    if case == "Argo":
        return (
            DeepKernelNNGP.NNMean,
            [nfeatures, [8, 8, 8], dropout],
        )
    else:
        return (
            DeepKernelNNGP.ConstMean,
            [0.0],
        )


def init_DeepKernelNNGP(d, device=None, case=None, dropout=0.0):
    if device is None:
        device = torch.device("cpu")
    input_trans_type = 'locs_lastloc'
    nfeatures = DeepKernelNNGP.input_transformed_dim(d, input_trans_type)
    MeanClass, mean_class_init = _get_mean_config(case, d, dropout)
    nn_size_configs = {
        "small": {"latent_dim": 32, "dim_middle": 64},
        "big": {"latent_dim": 64, "dim_middle": 128},
        "Argo": {"latent_dim": 16, "dim_middle": 16},
    }

    if case is None:
        nn_case = "big" if device.type == "cuda" else "small"
    else:
        if case not in nn_size_configs:
            raise ValueError(
                f"Invalid case={case!r}. Expected one of {list(nn_size_configs)} or None."
            )
        nn_case = case
    CovClass = DeepKernelNNGP.NNKernel
    latent_dim = nn_size_configs[nn_case]["latent_dim"]
    dim_middle = nn_size_configs[nn_case]["dim_middle"]

    nlayer_middle_rho = 3
    nlayer_middle_phi = 3
    nlayer_middle_rho1 = 3
    nlayer_middle_rho2 = 3
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

def init_VGP(d, device=None, case=None, dropout=0.0):
    if device is None:
        device = torch.device("cpu")

    nfeatures = d  # adjust if you later use input transforms
    MeanClass, mean_class_init = _get_mean_config(case, nfeatures, dropout)

    CovClass = DeepKernelNNGP.MyMaternKernel
    cov_class_init = [0.5, [0.1 for _ in range(d)], 1.5, 0.01]

    model_specs = {
        "MeanClass": MeanClass.__name__,
        "mean_class_init": mean_class_init,
        "CovClass": CovClass.__name__,
        "cov_class_init": cov_class_init,
    }

    model = DeepKernelNNGP.GPVecchia(
        MeanClass, CovClass, mean_class_init, cov_class_init
    )

    return model, model_specs

def init_VGP_SM(d, device=None, case=None, dropout=0.0):
    if device is None:
        device = torch.device("cpu")

    nfeatures = d
    MeanClass, mean_class_init = _get_mean_config(case, nfeatures, dropout)
    CovClass = DeepKernelNNGP.SpectralMixtureKernel
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

def init_VGP_Wilson2015Deep(d, device=None, case=None, dropout=0.0):
    if device is None:
        device = torch.device("cpu")

    nfeatures = d
    MeanClass, mean_class_init = _get_mean_config(case, nfeatures, dropout)
    
    nn_size_configs = {
        "default": [nfeatures, 1000, 1000, 500, 50, 2],
        "Argo": [nfeatures, 500, 500, 250, 50, 2],
    }
    if case is None:
        nn_case = "default"
    else:
        if case not in nn_size_configs:
            raise ValueError(
                f"Invalid case={case!r}. Expected one of {list(nn_size_configs)} or None."
            )
        nn_case = case
    CovClass = DeepKernelNNGP.Wilson2015Deep
    cov_class_init = [nn_size_configs[nn_case]] + [[0.1] * 6] + \
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