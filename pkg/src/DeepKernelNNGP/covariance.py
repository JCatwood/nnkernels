import torch
import gpytorch

class MyMaternKernel(torch.nn.Module):
    """
    Matérn covariance with ARD lengthscales, marginal scale, and nugget.

    Parameters
    ----------
    scale : float
        Marginal standard deviation parameter.
    lengthscale : sequence of float
        Positive ARD lengthscales, one per input dimension.
    nu : float
        Matérn smoothness parameter.
    nugget : float
        Positive nugget added to the diagonal.
    """
    def __init__(self, scale, lengthscale, nu, nugget):
        super().__init__()

        lengthscale_t = torch.as_tensor(lengthscale, dtype=torch.get_default_dtype())

        assert scale > 0
        assert nugget > 0
        assert torch.all(lengthscale_t > 0)

        self.gpt_matern = gpytorch.kernels.MaternKernel(nu=nu)
        self.gpt_matern.lengthscale = 1.0

        for param in self.gpt_matern.parameters():
            param.requires_grad = False

        self.d = len(lengthscale)
        self.raw_scale = torch.nn.Parameter(
            torch.log(torch.as_tensor(scale, dtype=torch.get_default_dtype()))
        )
        self.raw_lengthscale = torch.nn.Parameter(torch.log(lengthscale_t))
        self.raw_nugget = torch.nn.Parameter(
            torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype()))
        )
    
    def forward(self, x, **params):
        x_scaled = x / self.lengthscale
        covmat_parent = self.gpt_matern(x_scaled, x_scaled, **params)
        # If this returns a LazyTensor/LinearOperator and you want dense output:
        covmat_parent = covmat_parent.to_dense() if hasattr(covmat_parent, "to_dense") else covmat_parent
        n = covmat_parent.shape[-1]
        covmat = covmat_parent * (self.scale ** 2) + \
            torch.eye(n, device=x.device, dtype=x.dtype) * self.nugget
        return covmat

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        with torch.no_grad():
            self.raw_nugget.copy_(torch.log(
                torch.as_tensor(new_nugget, device=self.raw_nugget.device, dtype=self.raw_nugget.dtype)))
    
    @property
    def scale(self):
        return torch.exp(self.raw_scale)
    
    @scale.setter
    def scale(self, new_scale):
        with torch.no_grad():
             self.raw_scale.copy_(torch.log(
                torch.as_tensor(new_scale, device=self.raw_scale.device, dtype=self.raw_scale.dtype)))
    
    @property
    def lengthscale(self):
        return torch.exp(self.raw_lengthscale)
    
    @lengthscale.setter
    def lengthscale(self, new_lengthscale):
        with torch.no_grad():
            self.raw_lengthscale.copy_(torch.log(
                torch.as_tensor(new_lengthscale, device=self.raw_lengthscale.device, 
                                dtype=self.raw_lengthscale.dtype)))


class MyNSKernel_Scale(torch.nn.Module):
    """
    MyMaternKernel kernel with varying scale: 
    scale = exp(beta0 + beta1 * x[..., -1] + beta2 * x[..., -1]**2)
    """
    def __init__(self, beta0, beta1, beta2, lengthscale, nu, nugget, **kwargs):
        super().__init__()
        self.gpt_matern = gpytorch.kernels.MaternKernel(nu=nu)
        self.gpt_matern.lengthscale = lengthscale
        self.beta0 = torch.nn.Parameter(torch.as_tensor(beta0, dtype=torch.get_default_dtype()))
        self.beta1 = torch.nn.Parameter(torch.as_tensor(beta1, dtype=torch.get_default_dtype()))
        self.beta2 = torch.nn.Parameter(torch.as_tensor(beta2, dtype=torch.get_default_dtype()))
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype())))
    
    def forward(self, x, **params):
        if x.dim() == 2:
            x_view = x.unsqueeze(0)
        else:
            x_view = x
        sigma = torch.exp(self.beta0 + \
            self.beta1 * x_view[..., -1] + \
            self.beta2 * x_view[..., -1]**2)
        covmat_parent = self.gpt_matern(x_view, x_view, **params)
        covmat_parent = covmat_parent.to_dense() if hasattr(covmat_parent, "to_dense") else covmat_parent
        covmat_scaled = covmat_parent * sigma.unsqueeze(-1) * sigma.unsqueeze(1)
        n = covmat_scaled.shape[-1]
        covmat = covmat_scaled + torch.diag_embed(sigma**2 * self.nugget)
        if x.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat
    
    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        with torch.no_grad():
            self.raw_nugget.copy_(
                torch.log(torch.as_tensor(new_nugget, device=self.raw_nugget.device, dtype=self.raw_nugget.dtype))
            )
    
class MyNSKernel_Lengthscale(torch.nn.Module):
    """
    Exponential covariance kernel with varying lengthscale: 
    lengthscale = exp(beta0 + beta1 * sin(sum(x, dim=-1) * 3pi) + beta2 * cos(sum(x, dim=-1) * 2pi))
    Refer to Eq. 2-4 of https://arxiv.org/pdf/2208.07431
    """
    def __init__(self, beta0, beta1, beta2, scale, nugget):
        super().__init__()
        self.beta0 = torch.nn.Parameter(torch.as_tensor(beta0, dtype=torch.get_default_dtype()))
        self.beta1 = torch.nn.Parameter(torch.as_tensor(beta1, dtype=torch.get_default_dtype()))
        self.beta2 = torch.nn.Parameter(torch.as_tensor(beta2, dtype=torch.get_default_dtype()))
        self.raw_scale = torch.nn.Parameter(torch.log(torch.as_tensor(scale, dtype=torch.get_default_dtype())))
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype())))
    
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
        covmat = covmat_raw * (self.scale ** 2) + torch.eye(n, device=x_view.device, dtype=x_view.dtype) * self.nugget
        if x.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat
    
    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)
    
    @nugget.setter
    def nugget(self, new_nugget):
        with torch.no_grad():
            self.raw_nugget.copy_(
                torch.log(torch.as_tensor(new_nugget, device=self.raw_nugget.device, dtype=self.raw_nugget.dtype))
            )
    
    @property
    def scale(self):
        return torch.exp(self.raw_scale)
    
    @scale.setter
    def scale(self, new_scale):
        with torch.no_grad():
            self.raw_scale.copy_(
                torch.log(torch.as_tensor(new_scale, device=self.raw_scale.device, dtype=self.raw_scale.dtype))
            )
    

class MyNSKernel_Kron(torch.nn.Module):
    """
    Separable kernel K = K1 \\times K2 ... Kd ... 
    K1, K2, ... are the same
    """
    def __init__(self, lengthscale, nu, nugget, **kwargs):
        super().__init__(**kwargs)
        
        # Use constraints instead of assertions
        self.raw_nugget = torch.nn.Parameter(torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype())))

        # The base stationary kernel
        # We use a MaternKernel with a lengthscale
        self.base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
        self.base_kernel.lengthscale = lengthscale

    @property
    def nugget(self): return torch.exp(self.raw_nugget)

    def forward(self, x, **params):
        x1 = x
        x2 = x
        # 1. Compute the Stationary Kronecker Product
        res = None
        for d in range(x1.size(-1)):
            # Explicitly pass x2 to the base kernel
            k_part = self.base_kernel(x1[..., d:d+1], x2[..., d:d+1])
            res = k_part if res is None else res * k_part
        res_evaluated = res.to_dense() if hasattr(res, "to_dense") else res
        
        # 4. Add Nugget (jitter)
        n = x1.size(-2)
        res_evaluated = res_evaluated + torch.eye(n, device=x1.device, dtype=x1.dtype) * self.nugget
             
        return res_evaluated

class LinearKernel(torch.nn.Module):
    """
    Linear kernel K(x_i, x_j) = x_i^T x_j, with diagonal nugget.
    """
    def __init__(self, nugget, **kwargs):
        super().__init__(**kwargs)
        self.raw_nugget = torch.nn.Parameter(
            torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype()))
        )

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)

    def forward(self, x, **params):
        if x.dim() == 2:
            x_view = x.unsqueeze(0)
        else:
            x_view = x

        covmat = x_view @ x_view.transpose(-1, -2)

        n = x_view.size(-2)
        covmat = covmat + torch.eye(n, device=x_view.device, dtype=x_view.dtype) * self.nugget

        if x.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat