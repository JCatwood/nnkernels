import torch
from torch import nn
import gpytorch

def _build_block(sizes, dropout=0.0, normalize=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Apply activation and norm to everything EXCEPT the final projection
        if i < len(sizes) - 2:
            if normalize:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class PermInvarClass(torch.nn.Module):
    """
    TBD
    """
    def __init__(self, NNDT_size_seq, NNTG_size_seq, dropout=0.0):
        super().__init__()
        self.DT = _build_block(NNDT_size_seq, dropout, normalize=False)
        self.TG = _build_block(NNTG_size_seq, dropout, normalize=True)

    
    def forward(self, X, *args, **kwargs):
        X_after_DT = self.DT(X)
        X_after_sum = torch.sum(X_after_DT, dim=-2)
        target = self.TG(X_after_sum)
        return target

class PermPreserveClass(torch.nn.Module):
    """
    TBD
    """
    def __init__(self, phi_sz_seq, rho1_sz_seq, rho2_sz_seq, dropout=0.0, *args, **kwargs):
        super().__init__()
        self.phi = _build_block(phi_sz_seq, dropout, normalize=False)
        self.rho2 = _build_block(rho2_sz_seq, dropout, normalize=True)
        self.rho1 = _build_block(rho1_sz_seq, dropout, normalize=True)

    
    def forward(self, X, *args, **kwargs):
        """
        X: [*, m, d]
        """
        phi = self.phi(X) # [*, m, d_phi]
        phi_sum = torch.sum(phi, dim=-2, keepdim=True) # [*, 1, d_phi]
        phi_one2n = phi_sum - phi # [*, m, d_phi]
        rho2 = self.rho2(phi_one2n) # [*, m, d_rho2]
        rho1_input = torch.cat((phi, rho2), dim=-1) # [*, m, d_phi + d_rho2]
        rho1 = self.rho1(rho1_input) # [*, m, d_rho1]
        return rho1

class MyMaternKernel(gpytorch.kernels.MaternKernel):
    def __init__(self, scale, lengthscale, nu, nugget, **kwargs):
        super().__init__(nu, **kwargs)
        self.raw_scale = torch.nn.Parameter(torch.log(torch.tensor(scale)))
        self.lengthscale = lengthscale
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.tensor(nugget)))
    
    def forward(self, x1, x2, **params):
        covmat_parent = super().forward(x1, x2, **params)
        n = covmat_parent.shape[-1]
        covmat = covmat_parent * (self.scale ** 2) + torch.eye(n).to(self.device) * self.nugget
        return covmat

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        self.raw_nugget.copy_(torch.log(torch.tensor(new_nugget)))
    
    @property
    def scale(self):
        return torch.exp(self.raw_scale)
    
    @scale.setter
    def scale(self, new_scale):
        self.raw_scale.copy_(torch.log(torch.tensor(new_scale)))
    
    def __call__(self, x1, x2=None, **params):
        if x2 is None:
            x2 = x1
        return self.forward(x1, x2, **params)


class GPVecchia(torch.nn.Module):
    """
    The Vecchia approximation of a zero-mean GP
    """
    def __init__(self, KernelCls, *args, **kwargs):
        super().__init__()
        self.kernel = KernelCls(*args, **kwargs)

    def forward(self, locs_batch, y_batch, length=None):
        N = locs_batch.size(0)
        n_max = locs_batch.size(1)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat).to_dense()
        x = torch.linalg.solve_triangular(L, y_batch.reshape(N, n_max, 1), upper=False)
        L_zero_diag = L - L * torch.eye(n_max).unsqueeze(0).to(L.device)
        cond_mean_tmp = L_zero_diag @ x
        
        if isinstance(length, int) or length is None:
            cond_sd = L[:, -1, -1].unsqueeze(-1)
            cond_mean = cond_mean_tmp[:, -1, 0].unsqueeze(-1)
        else:
            cond_sd = L[torch.arange(N), length - 1, length - 1].unsqueeze(-1)
            cond_mean = cond_mean_tmp[torch.arange(N), length - 1, 0].unsqueeze(-1)
        
        return cond_mean, cond_sd
    
    def krig_coeff(self, locs_batch):
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)
        covmat_inv = torch.cholesky_inverse(L, upper=False)
        invchol_lastcol = covmat_inv[:, :, -1:] / \
            (covmat_inv[:, -1:, -1:] ** 0.5) # [N, m+1, 1]
        coeff = - invchol_lastcol[:, :-1, :] / invchol_lastcol[:, -1:, :] # [N, m, 1]
        return coeff
    
    def sample(self, locs_batch):
        N = locs_batch.size(0)
        m = locs_batch.size(1) - 1
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat) # [N, m+1, m+1]
        z = torch.randn([N, m+1, 1], dtype=locs_batch.dtype, device=locs_batch.device)
        return L @ z # [N, m+1, 1]


class MyNSKernel_Scale(gpytorch.kernels.MaternKernel):
    """
    MyMaternKernel kernel with varying scale: 
    scale = exp(beta0 + 1.25 * sin(x[..., 0] * beta1 * pi) + 1.25 * cos(x[..., 1] * beta2 * pi))
    """
    def __init__(self, beta0, beta1, beta2, lengthscale, nu, nugget, **kwargs):
        super().__init__(nu, **kwargs)
        self.beta0 = torch.nn.Parameter(torch.tensor(beta0))
        self.beta1 = torch.nn.Parameter(torch.tensor(beta1))
        self.beta2 = torch.nn.Parameter(torch.tensor(beta2))
        self.lengthscale = lengthscale
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.tensor(nugget)))
    
    def forward(self, x1, x2, **params):
        if x1.dim() == 2:
            x1_view = x1.unsqueeze(0)
            x2_view = x2.unsqueeze(0)
        else:
            x1_view = x1
            x2_view = x2
        sigma_x1 = torch.exp(self.beta0 + \
            1.25 * torch.sin(torch.sum(x1_view, dim=-1) * torch.pi * self.beta1) + \
            1.25 * torch.cos(torch.sum(x1_view, dim=-1) * torch.pi * self.beta2))
        sigma_x2 = torch.exp(self.beta0 + \
            1.25 * torch.sin(torch.sum(x2_view, dim=-1) * torch.pi * self.beta1) + \
            1.25 * torch.cos(torch.sum(x2_view, dim=-1) * torch.pi * self.beta2))
        covmat_parent = super().forward(x1_view, x2_view, **params)
        covmat_scaled = covmat_parent * sigma_x1.unsqueeze(-1) * sigma_x2.unsqueeze(1)
        n = covmat_scaled.shape[-1]
        covmat = covmat_scaled + torch.eye(n).to(self.device) * self.nugget
        if x1.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat
    
    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        self.raw_nugget.copy_(torch.log(torch.tensor(new_nugget)))

    def __call__(self, x1, x2=None, **params):
        if x2 is None:
            x2 = x1
        return self.forward(x1, x2, **params)
    
class MyNSKernel_Lengthscale(torch.nn.Module):
    """
    Exponential covariance kernel with varying lengthscale: 
    lengthscale = exp(beta0 + beta1 * sin(sum(x, dim=-1) * 3pi) + beta2 * cos(sum(x, dim=-1) * 2pi))
    Refer to Eq. 2-4 of https://arxiv.org/pdf/2208.07431
    """
    def __init__(self, beta0, beta1, beta2, scale, nugget):
        super().__init__()
        self.beta0 = torch.nn.Parameter(torch.tensor(beta0))
        self.beta1 = torch.nn.Parameter(torch.tensor(beta1))
        self.beta2 = torch.nn.Parameter(torch.tensor(beta2))
        self.raw_scale = torch.nn.Parameter(torch.log(torch.tensor(scale)))
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.tensor(nugget)))
    
    def forward(self, x, **params):
        if x.dim() == 2:
            x_view = x.unsqueeze(0)
        else:
            x_view = x
        d = x_view.size(-1)
        lengthscale_x = torch.exp(self.beta0 + \
            self.beta1 * torch.sin(torch.sum(x_view, dim=-1) * torch.pi * 3) + \
            self.beta2 * torch.cos(torch.sum(x_view, dim=-1) * torch.pi * 2))
        lengthscale_x1 = lengthscale_x.unsqueeze(-1)
        lengthscale_x2 = lengthscale_x.unsqueeze(1)
        c = ((lengthscale_x1 ** 0.25) * (lengthscale_x2 ** 0.25) / 
             (((lengthscale_x1 + lengthscale_x2) / 2.0) ** 0.5)) ** d
        q = torch.cdist(x_view, x_view) / (((lengthscale_x1 + lengthscale_x2) / 2.0) ** 0.5)
        covmat_raw = torch.exp(-q) * c
        n = covmat_raw.shape[-1]
        covmat = covmat_raw * (self.scale ** 2) + torch.eye(n).to(x_view.device) * self.nugget
        if x.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat
    
    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        self.raw_nugget.copy_(torch.log(torch.tensor(new_nugget)))
    
    @property
    def scale(self):
        return torch.exp(self.raw_scale)
    
    @scale.setter
    def scale(self, new_scale):
        self.raw_scale.copy_(torch.log(torch.tensor(new_scale)))

    def __call__(self, x, **params):
        return self.forward(x, **params)
