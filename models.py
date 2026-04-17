import torch 
import DeepKernelNNGP

def init_DeepKernelNNGP(d, dropout=0.0, device=torch.device("cpu")):
    MeanClass = DeepKernelNNGP.ConstMean
    CovClass = DeepKernelNNGP.NNKernel
    input_trans_type = 'locs_lastloc'
    nfeatures = DeepKernelNNGP.input_transformed_dim(d, input_trans_type)
    if device.type == "cuda":
        latent_dim = 64
        dim_middle = 128
    else: # device == torch.device('cpu')
        latent_dim = 32
        dim_middle = 64
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