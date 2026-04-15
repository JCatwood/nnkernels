import torch
from .input_transform import input_transform

class DeepKernelNNGP(torch.nn.Module):
    """
    The Vecchia approximation of a GP
    """
    def __init__(self, MeanClass, KernelClass, mean_init_parms, kernel_init_parms, input_trans='locs_lastloc', use_trans_for_mean=False):
        super().__init__()
        self.mean = MeanClass(*mean_init_parms)
        self.kernel = KernelClass(*kernel_init_parms)
        self.input_trans_type = input_trans
        self.use_trans_for_mean = use_trans_for_mean
        
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
        locs_batch_trans = input_transform(locs_batch, type=self.input_trans_type)
        if self.use_trans_for_mean:
            mean = self.mean(locs_batch_trans) # B, m + 1, 1
        else:
            mean = self.mean(locs_batch) # B, m + 1, 1
        y_batch_demean = y_batch - mean
        kernel_input = locs_batch_trans[:, :-1, :]
        krig_coeff, cond_sd = self.kernel(kernel_input)
        cond_mean = torch.sum(krig_coeff * y_batch_demean[:, :-1, :], dim=1, keepdim=True) + mean[:, -1:, :]
        return cond_mean, cond_sd
    
    def forward(self, locs_batch, y_batch):
        """
        Alias for cond_mean_sd
        """
        return self.cond_mean_sd(locs_batch, y_batch)
    