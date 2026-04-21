import torch
import gpytorch
from torch import nn
import math
from .NNkernel import _build_block

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


class PeriodicKernel(torch.nn.Module):
    """
    Periodic kernel

        K(x_i, x_j) = scale^2 * exp(
            -2 * sum_q sin^2(pi * (x_iq - x_jq) / period) / lengthscale
        )

    with diagonal nugget added.
    """
    def __init__(self, scale, period, lengthscale, nugget, **kwargs):
        super().__init__(**kwargs)

        if scale <= 0 or period <= 0 or lengthscale <= 0 or nugget <= 0:
            raise ValueError("scale, period, lengthscale, and nugget must be positive")

        self.raw_scale = torch.nn.Parameter(
            torch.log(torch.as_tensor(scale, dtype=torch.get_default_dtype()))
        )
        self.raw_period = torch.nn.Parameter(
            torch.log(torch.as_tensor(period, dtype=torch.get_default_dtype()))
        )
        self.raw_lengthscale = torch.nn.Parameter(
            torch.log(torch.as_tensor(lengthscale, dtype=torch.get_default_dtype()))
        )
        self.raw_nugget = torch.nn.Parameter(
            torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype()))
        )

    @property
    def scale(self):
        return torch.exp(self.raw_scale)

    @property
    def period(self):
        return torch.exp(self.raw_period)

    @property
    def lengthscale(self):
        return torch.exp(self.raw_lengthscale)

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)

    def forward(self, x, **params):
        if x.dim() == 2:
            x_view = x.unsqueeze(0)
        else:
            x_view = x

        diff = x_view.unsqueeze(-2) - x_view.unsqueeze(-3)   # [B, n, n, d]
        sin2 = torch.sin(torch.pi * diff / self.period) ** 2
        covmat = torch.exp(-2.0 * sin2.sum(dim=-1) / self.lengthscale) * (self.scale ** 2)

        n = x_view.size(-2)
        covmat = covmat + torch.eye(n, device=x_view.device, dtype=x_view.dtype) * self.nugget

        if x.dim() == 2:
            return covmat.squeeze(0)
        else:
            return covmat

def _generate_magic_square(d):
    if d % 2 == 0:
        raise ValueError("This specific implementation is for odd dimensions only.")

    # Initialize a d x d matrix with zeros
    magic_square = torch.zeros((d, d))

    # Starting position for 1
    row, col = 0, d // 2

    for num in range(1, d**2 + 1):
        magic_square[row, col] = num
        
        # Calculate next position: up one, right one
        new_row, new_col = (row - 1) % d, (col + 1) % d
        
        # If the cell is already filled, move down instead
        if magic_square[new_row, new_col]:
            row = (row + 1) % d
        else:
            row, col = new_row, new_col

    return magic_square

class TransformedMaternKernel(torch.nn.Module):
    """
    Isotropic Matérn kernel with a learned domain transformation.

    The input coordinates are first transformed by a random matrix
    Parameters
    ----------
    d : int
        Input dimension.
    hidden_dim : int, default=64
        Hidden width of the domain transform network.
    scale : float, default=1.0
        Marginal standard deviation.
    lengthscale : float, default=0.2
        Isotropic lengthscale.
    nu : float, default=1.5
        Matérn smoothness parameter.
    nugget : float, default=1e-3
        Diagonal nugget.
    """

    def __init__(
        self,
        d: int, scale: float = 1.0, lengthscale: float = 0.1, 
        nu: float = 1.5, nugget: float = 1e-3,
    ):
        super().__init__()

        if scale <= 0:
            raise ValueError("scale must be positive")
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if nugget <= 0:
            raise ValueError("nugget must be positive")

        self.d = d
        self.transformer = torch.nn.Parameter(_generate_magic_square(d) / (d * (d**2 + 1) / 2))

        # Use a base Matérn kernel with unit lengthscale after manual scaling
        self.base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
        self.base_kernel.lengthscale = lengthscale

        self.raw_scale = torch.nn.Parameter(
            torch.log(torch.as_tensor(scale, dtype=torch.get_default_dtype()))
        )
        self.raw_nugget = torch.nn.Parameter(
            torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype()))
        )

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.raw_scale)

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.base_kernel.lengthscale

    @property
    def nugget(self) -> torch.Tensor:
        return torch.exp(self.raw_nugget)

    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [n, d] or [B, n, d]

        Returns
        -------
        K : torch.Tensor
            Shape [n, n] or [B, n, n]
        """
        z = x @ self.transformer

        K = self.base_kernel(z, z)
        K = K.to_dense() if hasattr(K, "to_dense") else K

        n = z.shape[-2]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        if z.dim() == 3:
            eye = eye.unsqueeze(0)

        K = (self.scale ** 2) * K + self.nugget * eye
        return K
    
class _SpectralMixtureBaseKernel(nn.Module):
    """
    One Wilson-Adams (2013) spectral mixture component.

    k_q(x, x') = w_q * prod_p exp(-2 pi^2 (x_p - x'_p)^2 v_q^(p))
                        * cos(2 pi (x_p - x'_p) mu_q^(p))

    Parameters
    ----------
    weight : float
        Positive component weight.
    mean : array-like of shape [d]
        Spectral mean (frequency) for each input dimension.
    variance : array-like of shape [d]
        Spectral variance for each input dimension.
    """

    def __init__(self, weight, mean, variance):
        super().__init__()

        mean_t = torch.as_tensor(mean, dtype=torch.get_default_dtype())
        var_t = torch.as_tensor(variance, dtype=torch.get_default_dtype())

        if mean_t.ndim != 1:
            raise ValueError("mean must be a 1D tensor/list of shape [d]")
        if var_t.ndim != 1:
            raise ValueError("variance must be a 1D tensor/list of shape [d]")
        if mean_t.shape != var_t.shape:
            raise ValueError("mean and variance must have the same shape")
        if weight <= 0:
            raise ValueError("weight must be positive")
        if torch.any(var_t <= 0):
            raise ValueError("all variances must be positive")

        self.d = mean_t.numel()
        self.raw_weight = nn.Parameter(torch.log(torch.as_tensor(weight, dtype=torch.get_default_dtype())))
        self.raw_mean = nn.Parameter(torch.log(mean_t.clamp_min(1e-12)))
        self.raw_variance = nn.Parameter(torch.log(var_t))

    @property
    def weight(self):
        return torch.exp(self.raw_weight)

    @property
    def mean(self):
        return torch.exp(self.raw_mean)

    @property
    def variance(self):
        return torch.exp(self.raw_variance)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape [n, d] or [B, n, d]

        Returns
        -------
        K : Tensor
            Shape [n, n] or [B, n, n]
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected last dimension {self.d}, got {x.shape[-1]}")

        if x.dim() == 2:
            x_view = x.unsqueeze(0)  # [1, n, d]
            squeeze_out = True
        elif x.dim() == 3:
            x_view = x
            squeeze_out = False
        else:
            raise ValueError("x must have shape [n, d] or [B, n, d]")

        diff = x_view.unsqueeze(-2) - x_view.unsqueeze(-3)  # [B, n, n, d]

        exp_term = torch.exp(
            -2.0 * (math.pi ** 2) * (diff ** 2) * self.variance.view(1, 1, 1, self.d)
        )
        cos_term = torch.cos(
            2.0 * math.pi * diff * self.mean.view(1, 1, 1, self.d)
        )

        K = self.weight * (exp_term * cos_term).prod(dim=-1)

        return K.squeeze(0) if squeeze_out else K


class SpectralMixtureKernel(nn.Module):
    """
    Full Wilson-Adams (2013) spectral mixture kernel:
        k(x, x') = sum_{q=1}^Q k_q(x, x')

    Parameters
    ----------
    weights : array-like of shape [Q]
        Positive mixture weights.
    means : array-like of shape [Q, d]
        Spectral means per component and dimension.
    variances : array-like of shape [Q, d]
        Spectral variances per component and dimension.
    nugget : float, default=1e-6
        Positive diagonal nugget.
    """

    def __init__(self, weights, means, variances, nugget=1e-6):
        super().__init__()

        weights_t = torch.as_tensor(weights, dtype=torch.get_default_dtype())
        means_t = torch.as_tensor(means, dtype=torch.get_default_dtype())
        variances_t = torch.as_tensor(variances, dtype=torch.get_default_dtype())

        if weights_t.ndim != 1:
            raise ValueError("weights must have shape [Q]")
        if means_t.ndim != 2:
            raise ValueError("means must have shape [Q, d]")
        if variances_t.ndim != 2:
            raise ValueError("variances must have shape [Q, d]")
        if means_t.shape != variances_t.shape:
            raise ValueError("means and variances must have the same shape [Q, d]")
        if means_t.shape[0] != weights_t.shape[0]:
            raise ValueError("weights, means, and variances must agree on Q")
        if nugget <= 0:
            raise ValueError("nugget must be positive")
        if torch.any(weights_t <= 0):
            raise ValueError("all weights must be positive")
        if torch.any(variances_t <= 0):
            raise ValueError("all variances must be positive")
        if torch.any(means_t < 0):
            raise ValueError("all means must be nonnegative")

        self.Q = weights_t.numel()
        self.d = means_t.shape[1]
        self.raw_nugget = nn.Parameter(torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype())))

        self.components = nn.ModuleList(
            [
                _SpectralMixtureBaseKernel(
                    weight=weights_t[q].item(),
                    mean=means_t[q],
                    variance=variances_t[q],
                )
                for q in range(self.Q)
            ]
        )

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape [n, d] or [B, n, d]

        Returns
        -------
        K : Tensor
            Shape [n, n] or [B, n, n]
        """
        K = None
        for comp in self.components:
            Kq = comp(x)
            K = Kq if K is None else K + Kq

        n = x.shape[-2]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        if x.dim() == 3:
            eye = eye.unsqueeze(0)

        return K + self.nugget * eye

class Wilson2015Deep(torch.nn.Module):
    def __init__(self, NN_size, weights, means, variances, nugget=1e-6):
        super().__init__()
        self.transformer = _build_block(NN_size)
        self.sm_kernel = SpectralMixtureKernel(weights, means, variances, nugget)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape [n, d] or [B, n, d]

        Returns
        -------
        K : Tensor
            Shape [n, n] or [B, n, n]
        """
        x_trans = self.transformer(x)
        return self.sm_kernel(x_trans)
        