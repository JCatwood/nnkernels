import torch
import math

class GPVecchia(torch.nn.Module):
    """
    The Vecchia approximation of a GP
    """
    def __init__(self, MeanClass, KernelClass, mean_init_parms, kernel_init_parms):
        super().__init__()
        self.mean = MeanClass(*mean_init_parms)
        self.kernel = KernelClass(*kernel_init_parms)
        
    def cond_mean_sd(self, locs_batch, y_batch):
        """
        Compute Vecchia/nearest-neighbor type conditional mean and std. Return B batches of univariate conditional mean and std corresponding to y_batch[..., -1, 0] | y_batch[..., :-1, 0]

        Parameters:
            locs_batch: of shape [B, m + 1, d]
            y_batch: of shape [B, m + 1, 1]

        Returns:
            cond_mean: of shape [B, 1, 1]
            cond_sd: of shape [B, 1, 1]
        """
        covmat = self.kernel(locs_batch) # B, m + 1, m + 1
        mean = self.mean(locs_batch) # B, m + 1, 1
        y_batch_demean = y_batch - mean
        L = torch.linalg.cholesky(covmat)
        x = torch.linalg.solve_triangular(L, y_batch_demean, upper=False)
        cond_mean = L[:, -1:, :-1] @ x[:, :-1, :] + mean[:, -1:, :] # B, 1, 1
        cond_sd = L[:, -1:, -1:] # B, 1, 1
        return cond_mean, cond_sd

    def krig_coeff(self, locs_batch):
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)
        covmat_inv = torch.cholesky_inverse(L, upper=False)
        invchol_lastcol = covmat_inv[:, :, -1:] / \
            (covmat_inv[:, -1:, -1:] ** 0.5) # [N, m+1, 1]
        coeff = - invchol_lastcol[:, :-1, :] / invchol_lastcol[:, -1:, :] # [N, m, 1]
        return coeff
    
    def forward(self, locs_batch, y_batch):
        """
        Alias for cond_mean_sd
        """
        return self.cond_mean_sd(locs_batch, y_batch)
    
    def sample(self, locs_batch):
        N = locs_batch.size(0)
        m = locs_batch.size(1) - 1
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat) # [N, m+1, m+1]
        z = torch.randn([N, m+1, 1], dtype=locs_batch.dtype, device=locs_batch.device)
        mean = self.mean(locs_batch)
        return L @ z + mean # [N, m+1, 1]