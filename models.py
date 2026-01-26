import torch
from torch import nn
import gpytorch

class NNDT_Sum_NNTG(torch.nn.Module):
    """
    Given a tensor of dim [*, n, d], representing batches of n locs embeded in R^d, 
    if output == 'scalar'
    the forward propogation first transforms each d-dimensional coordinate vector (last dimension), then aggregate (sum) across the n locs (2nd last dimension), finally pass the previous result though another NN to predict the target
    otherwise
    the aggregated vector across n locs is concatenated with each of the transformed vector and pass through another NN as different batches. The output will be of shape [*, n, 1], with correspondence to the input
    """
    def __init__(self, NNDT_size_seq, NNTG_size_seq, dropout=0.0, aggregate_mtd='sum', 
                 output="scalar"):
        super().__init__()
        layers = []
        for i in range(len(NNDT_size_seq) - 1):
            layers.append(nn.Linear(NNDT_size_seq[i], NNDT_size_seq[i + 1]))
            if i < len(NNDT_size_seq) -  2:
                layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(p=dropout))
        self.DT = nn.Sequential(*layers)

        layers = []
        for i in range(len(NNTG_size_seq) - 1):
            layers.append(nn.Linear(NNTG_size_seq[i], NNTG_size_seq[i + 1]))
            if i < len(NNTG_size_seq) -  2:
                layers.append(nn.ReLU())
        self.TG = nn.Sequential(*layers)

        assert aggregate_mtd in ['sum', 'mean'], "aggregate_mtd must be one of ['sum', 'mean']"
        self.aggregate_mtd = aggregate_mtd
        assert output in ['scalar', 'vector'], "output must be one of ['scalar', 'vector']"
        self.output = output

    
    def forward(self, X, length=None):
        X_after_DT = self.DT(X)
        n_batch, length_max = X.size(0), X.size(1)
        if isinstance(length, int) or length is None:
            if self.aggregate_mtd == "sum":
                X_after_sum = torch.sum(X_after_DT, dim = -2)
            elif self.aggregate_mtd == "mean":
                X_after_sum = torch.sum(X_after_DT, dim = -2) / \
                    torch.tensor(length_max, dtype=X_after_DT.dtype)
        else:
            assert len(length) == n_batch
            mask_float = torch.zeros_like(X_after_DT)
            n = X.size(1)
            mask_bool = torch.arange(n).reshape(1, n) < length.reshape(-1, 1) # n_batch X n
            mask_float[mask_bool, :] = 1.0
            if self.aggregate_mtd == "sum":
                X_after_sum = torch.sum(X_after_DT * mask_float, dim = -2)
            elif self.aggregate_mtd == "mean":
                length_float = length.to(X_after_DT.dtype).unsqueeze(-1)
                X_after_sum = torch.sum(X_after_DT * mask_float, dim = -2) / length_float
        if self.output == "scalar":
            target = self.TG(X_after_sum).squeeze()
        else:
            X_after_sum_view = X_after_sum.unsqueeze(-2).expand(*X_after_DT.shape)
            input_cat = torch.cat((X_after_sum_view, X_after_DT), dim=-1)
            target = self.TG(input_cat)
        return target

class NNDT2_Sum_NNTG(torch.nn.Module):
    """
    Given a tensor of dim [*, n, d], representing batches of n locs embeded in R^d, 
    the forward propogation first transforms each d-dimensional coordinate vector (last dimension) using DT1, then aggregate (sum) across the n locs (2nd last dimension), the aggregated vector across n locs is concatenated with each of the transformed input using DT2, and pass through another NN (TG) as different batches. The output will be of shape [*, n, 1], with correspondence to the input, finally pass the previous result though another NN to predict the target
    """
    def __init__(self, NNDT1_size_seq, NNDT2_size_seq, NNTG_size_seq, *args, **kwargs):
        assert NNDT1_size_seq[-1] + NNDT2_size_seq[-1] == NNTG_size_seq[0], "size mismatch between DT1, DT2 and TG"
        super().__init__()
        layers = []
        for i in range(len(NNDT1_size_seq) - 1):
            layers.append(nn.Linear(NNDT1_size_seq[i], NNDT1_size_seq[i + 1]))
            if i < len(NNDT1_size_seq) -  2:
                layers.append(nn.ReLU())
        self.DT1 = nn.Sequential(*layers)

        layers = []
        for i in range(len(NNDT2_size_seq) - 1):
            layers.append(nn.Linear(NNDT2_size_seq[i], NNDT2_size_seq[i + 1]))
            if i < len(NNDT2_size_seq) -  2:
                layers.append(nn.ReLU())
        self.DT2 = nn.Sequential(*layers)

        layers = []
        for i in range(len(NNTG_size_seq) - 1):
            layers.append(nn.Linear(NNTG_size_seq[i], NNTG_size_seq[i + 1]))
            if i < len(NNTG_size_seq) -  2:
                layers.append(nn.ReLU())
        self.TG = nn.Sequential(*layers)

    
    def forward(self, X, *args, **kwargs):
        X_after_DT1 = self.DT1(X)
        X_after_DT2 = self.DT2(X)
        X_DT1_reduce = torch.sum(X_after_DT1, dim=-2, keepdim=True).expand(-1, X.size(-2), -1)
        input_TG = torch.cat((X_DT1_reduce, X_after_DT2), dim=-1)
        target = self.TG(input_TG)
        return target
    
class LSTM_NNTG(torch.nn.Module):
    """
    Given a tensor X of dim [*, n, d], representing batches of n locs embeded in R^d, the
    forward propogation first passes X through a LSTM and then pass the last layer of the output to a NN to predict the target
    """
    def __init__(self, XDim, nHidden):
        super().__init__()
        self.n_hidden = nHidden
        self.n_feature = XDim
        self.lstm = torch.nn.LSTM(XDim, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, X, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(X)
        else:
            lstm_out, (h_out, c_out) = self.lstm(X, (h0, c0))
        out = self.hidden2pred(lstm_out[:, -1, :]).reshape((-1,))
        return out, h_out, c_out

class MyMaternKernel(gpytorch.kernels.MaternKernel):
    def __init__(self, scale, lengthscale, nu, nugget):
        super().__init__(nu)
        is_stationary = True
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
            cond_sd = L[:, -1, -1]
            cond_mean = cond_mean_tmp[:, -1, 0]
        else:
            cond_sd = L[torch.arange(N), length - 1, length - 1]
            cond_mean = cond_mean_tmp[torch.arange(N), length - 1, 0]
        
        return cond_mean, cond_sd

class MyNSKernel_Scale(gpytorch.kernels.MaternKernel):
    """
    MyMaternKernel kernel with varying scale: 
    scale = exp(beta0 + beta1 * sin(x * 2pi) + beta2 * cos(y * 2pi))
    """
    def __init__(self, beta0, beta1, beta2, lengthscale, nu, nugget):
        super().__init__(nu)
        is_stationary = False
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
        sigma_x1 = torch.exp(self.beta0 + self.beta1 * torch.sin(x1_view[:, :, 0] * torch.pi * 2) + 
                             self.beta2 * torch.cos(x1_view[:, :, 1] * torch.pi * 2))
        sigma_x2 = torch.exp(self.beta0 + self.beta1 * torch.sin(x2_view[:, :, 0] * torch.pi * 2) + 
                             self.beta2 * torch.cos(x2_view[:, :, 1] * torch.pi * 2))
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
    lengthscale = exp(beta0 + beta1 * sin(x * 2pi) + beta2 * cos(y * 2pi))
    Refer to Eq. 2-4 of https://arxiv.org/pdf/2208.07431
    """
    def __init__(self, beta0, beta1, beta2, scale, nugget):
        super().__init__()
        is_stationary = False
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
        lengthscale_x = torch.exp(self.beta0 + self.beta1 * torch.sin(x_view[:, :, 0] * torch.pi * 2) + 
                            self.beta2 * torch.cos(x_view[:, :, 1] * torch.pi * 2))
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
