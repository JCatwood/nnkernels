import torch
from torch import nn

class NNDT_Sum_NNTG(torch.nn.Module):
    """
    Given a tensor of dim [*, n, d], representing batches of n locs embeded in R^d, the
    forward propogation first transforms each d-dimensional coordinate vector (last dimension), then aggregate (sum) across the n locs (2nd last dimension), finally pass the previous result though another NN to predict the target
    """
    def __init__(self, NNDT_size_seq, NNTG_size_seq, dropout=0.0):
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
    
    def forward(self, X, length=None):
        X_after_DT = self.DT(X)
        if isinstance(length, int) or length is None:
            X_after_sum = torch.sum(X_after_DT, dim = -2)
        else:
            assert len(length) == X.size(0)
            mask_float = torch.zeros_like(X_after_DT)
            n = X.size(1)
            mask_bool = torch.arange(n).reshape(1, n) < length.reshape(-1, 1) # n_batch X n
            mask_float[mask_bool, :] = 1.0
            X_after_sum = torch.sum(X_after_DT * mask_float, dim = -2)
            
        target = self.TG(X_after_sum).squeeze()
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

class Kernel_Nugg(torch.nn.Module):
    """
    GPyTorch kernel with a nugget
    """
    def __init__(self, gpt_kernel, nugget):
        super().__init__()
        self.gpt_kernel = gpt_kernel
        self.raw_nugget = nn.Parameter(torch.log(torch.tensor(nugget)))
    
    def forward(self, *args, **kwargs):
        covmat = self.gpt_kernel(*args, **kwargs).evaluate()
        n = covmat.shape[-1]
        covmat_nug = covmat + torch.eye(n) * torch.exp(self.raw_nugget)
        return covmat_nug

class GPVecchia(torch.nn.Module):
    """
    The Vecchia approximation of a zero-mean GP
    """
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, locs_batch, y_batch, length=None):
        N = locs_batch.size(0)
        n_max = locs_batch.size(1)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat).to_dense()
        x = torch.linalg.solve_triangular(L, y_batch.unsqueeze(-1), upper=False)
        L_zero_diag = L - L * torch.eye(n_max).unsqueeze(0)
        cond_mean_tmp = L_zero_diag @ x
        
        if isinstance(length, int) or length is None:
            cond_sd = L[:, -1, -1]
            cond_mean = cond_mean_tmp[:, -1, 0]
        else:
            cond_sd = L[torch.arange(N), length - 1, length - 1]
            cond_mean = cond_mean_tmp[:, length - 1, 0]
        
        return cond_mean, cond_sd
