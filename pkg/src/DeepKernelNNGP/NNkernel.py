import torch
from torch import nn


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
    def __init__(self, NNDT_size_seq, NNTG_size_seq, dropout=0.0):
        assert NNTG_size_seq[-1] == 1, "the last entry of NNTG_size_seq should be 1"
        super().__init__()
        self.DT = _build_block(NNDT_size_seq, dropout, normalize=False)
        self.TG = _build_block(NNTG_size_seq, dropout, normalize=True)
    
    def forward(self, X):
        X_after_DT = self.DT(X)
        X_after_sum = torch.sum(X_after_DT, dim=-2)
        target = self.TG(X_after_sum)
        return target

class PermPreserveClass(torch.nn.Module):
    def __init__(self, phi_sz_seq, rho1_sz_seq, rho2_sz_seq, dropout=0.0):
        assert rho1_sz_seq[-1] == 1, "the last entry of NNTG_size_seq should be 1"
        super().__init__()
        self.phi = _build_block(phi_sz_seq, dropout, normalize=False)
        self.rho2 = _build_block(rho2_sz_seq, dropout, normalize=True)
        self.rho1 = _build_block(rho1_sz_seq, dropout, normalize=True)

    def forward(self, X):
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

class VanillaCoeffMLP(torch.nn.Module):
    """Vanilla MLP for predicting all kriging coefficients jointly.

    The input has shape [B, m, d_in]. It is flattened to [B, m * d_in],
    and the output has shape [B, m, 1].

    Unlike PermPreserveClass, this architecture is not permutation
    preserving with respect to the conditioning locations.
    """

    def __init__(self, m, nfeatures, dim_middle, nlayer_middle, dropout=0.0):
        super().__init__()

        if m < 1:
            raise ValueError("m must be positive")

        self.m = int(m)
        self.nfeatures = int(nfeatures)

        size_seq = [m * nfeatures] + [dim_middle for _ in range(nlayer_middle)] + [m]
        self.network = _build_block(size_seq, dropout=dropout, normalize=True)

    def forward(self, X):
        """
        Parameters
        ----------
        X : torch.Tensor
            Tensor with shape [B, m, nfeatures].

        Returns
        -------
        torch.Tensor
            Kriging coefficients with shape [B, m, 1].
        """
        if X.shape[-2] != self.m:
            raise ValueError(
                f"VanillaCoeffMLP was initialized with m={self.m}, "
                f"but received an input with {X.shape[-2]} conditioning locations."
            )

        if X.shape[-1] != self.nfeatures:
            raise ValueError(
                f"VanillaCoeffMLP expected {self.nfeatures} features, "
                f"but received {X.shape[-1]}."
            )

        X_flat = X.reshape(*X.shape[:-2], self.m * self.nfeatures)
        coefficients = self.network(X_flat)

        return coefficients.unsqueeze(-1)

class NNKernel(torch.nn.Module):
    def __init__(self, nfeatures, dim_middle, dim_latent, nlayer_middle_rho, nlayer_middle_phi,
                 nlayer_middle_rho1, nlayer_middle_rho2, dropout=0.2):
        super().__init__()
        size_phi = [nfeatures,] + [dim_middle for i in range(nlayer_middle_phi)] + [dim_latent,]
        size_rho2 = [dim_latent,] + [dim_middle for i in range(nlayer_middle_rho2)] + [dim_latent,]
        size_rho1 = [dim_latent + dim_latent,] + [dim_middle for i in range(nlayer_middle_rho1)] + [1,]
        size_rho = [dim_latent,] + [dim_middle for i in range(nlayer_middle_rho)] + [1,]
        self.model_coeff = PermPreserveClass(size_phi, size_rho1, size_rho2, dropout=dropout)
        self.model_sd = PermInvarClass(size_phi, size_rho, dropout=dropout)
    
    def forward(self, locs_trans_batch):
        """
        Return kriging coefficient and conditional sd
        Parameters:
            locs_batch: of shape [B, m, d]
        Returns:
            krig_coeff: of shape [B, m, 1]
            cond_sd: of shape [B, 1, 1]
        """
        krig_coeff = self.model_coeff(locs_trans_batch)
        cond_sd = torch.exp(self.model_sd(locs_trans_batch)).unsqueeze(-1)
        return krig_coeff, cond_sd

class NNKernel_BL_Coef(torch.nn.Module):
    """NeuVec ablation with a vanilla MLP for kriging coefficients."""

    def __init__(self, m, nfeatures, dim_middle, dim_latent, nlayer_middle_rho, nlayer_middle_phi, 
                 nlayer_middle_coeff, dropout=0.2,):
        super().__init__()
        size_phi = [nfeatures] + [dim_middle for _ in range(nlayer_middle_phi)] + [dim_latent]
        size_rho = [dim_latent] + [dim_middle for _ in range(nlayer_middle_rho)] + [1]

        self.model_coeff = VanillaCoeffMLP(m=m, nfeatures=nfeatures, dim_middle=dim_middle, 
            nlayer_middle=nlayer_middle_coeff, dropout=dropout)
        self.model_sd = PermInvarClass(size_phi, size_rho, dropout=dropout)

    def forward(self, locs_trans_batch):
        """
        Parameters
        ----------
        locs_trans_batch : torch.Tensor
            Tensor with shape [B, m, nfeatures].

        Returns
        -------
        krig_coeff : torch.Tensor
            Tensor with shape [B, m, 1].
        cond_sd : torch.Tensor
            Tensor with shape [B, 1, 1].
        """
        krig_coeff = self.model_coeff(locs_trans_batch)
        cond_sd = torch.exp(self.model_sd(locs_trans_batch)).unsqueeze(-1)

        return krig_coeff, cond_sd