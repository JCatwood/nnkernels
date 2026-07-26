import torch 
import DeepKernelNNGP

def _get_mean_config(case, nfeatures, dropout=0.0):
    if case in {"Argo", "GHRSST"}:
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
        "GHRSST": {"latent_dim": 16, "dim_middle": 16},
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

def init_DeepKernelNNGP_BL_Coef(d, m, device=None, case=None, dropout=0.0):
    """Initialize the coefficient-network ablation of NeuVec."""
    if device is None:
        device = torch.device("cpu")
    input_trans_type = "locs_lastloc"
    nfeatures = DeepKernelNNGP.input_transformed_dim(d, input_trans_type)
    MeanClass, mean_class_init = _get_mean_config(case, d, dropout)
    nn_size_configs = {
        "small": {"latent_dim": 32, "dim_middle": 64},
        "big": {"latent_dim": 64, "dim_middle": 128},
        "Argo": {"latent_dim": 16, "dim_middle": 16},
        "GHRSST": {"latent_dim": 16, "dim_middle": 16},
    }

    if case is None:
        nn_case = "big" if device.type == "cuda" else "small"
    else:
        if case not in nn_size_configs:
            raise ValueError(
                f"Invalid case={case!r}. "
                f"Expected one of {list(nn_size_configs)} or None."
            )
        nn_case = case
    CovClass = DeepKernelNNGP.NNKernel_BL_Coef
    latent_dim = nn_size_configs[nn_case]["latent_dim"]
    dim_middle = nn_size_configs[nn_case]["dim_middle"]

    nlayer_middle_rho = 3
    nlayer_middle_phi = 3
    nlayer_middle_coeff = 3

    cov_class_init = [m, nfeatures, dim_middle, latent_dim, nlayer_middle_rho,
        nlayer_middle_phi, nlayer_middle_coeff, dropout]

    model_specs = {
        "MeanClass": MeanClass.__name__,
        "mean_class_init": mean_class_init,
        "CovClass": CovClass.__name__,
        "cov_class_init": cov_class_init,
        "input_trans_type": input_trans_type,
        "coefficient_architecture": "vanilla_mlp",
    }

    model = DeepKernelNNGP.DeepKernelNNGP(
        MeanClass,
        CovClass,
        mean_class_init,
        cov_class_init,
        input_trans_type,
    )

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
    CovClass = DeepKernelNNGP.Wilson2015Deep
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

def init_SPGP(d, m=30, device=None, case=None, dropout=0.0):
    """Initialize the sparse pseudo-input GP baseline."""
    if device is None:
        device = torch.device("cpu")
    MeanClass, mean_class_init = _get_mean_config(case, d, dropout,)
    model = DeepKernelNNGP.SPGP(
        MeanClass=MeanClass,
        mean_class_init=mean_class_init,
        d=d,
        n_pseudo=m,
        scale=0.5,
        lengthscale=0.1,
        nugget=0.01,
        jitter=1e-5,
    )

    model_specs = {
        "ModelClass": model.__class__.__name__,
        "MeanClass": MeanClass.__name__,
        "mean_class_init": mean_class_init,
        "n_pseudo": m,
        "kernel": "Matern32",
        "scale_init": 0.5,
        "lengthscale_init": 0.1,
        "nugget_init": 0.01,
        "jitter": 1e-5,
    }

    return model, model_specs