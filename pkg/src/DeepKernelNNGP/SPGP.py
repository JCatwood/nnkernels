import math

import torch


class SPGP(torch.nn.Module):
    """Sparse pseudo-input GP using a Matérn-3/2 parent covariance.

    The model uses the FITC/SPGP covariance

        Q_xx + diag(K_xx - diag(Q_xx)) + nugget * I,

    where Q_xx = K_xu K_uu^{-1} K_ux.  It follows the repository's
    conditional-prediction interface: each input batch contains conditioning
    locations followed by one target location.
    """

    def __init__(
        self,
        MeanClass,
        mean_class_init,
        d,
        n_pseudo=30,
        scale=0.5,
        lengthscale=0.1,
        nugget=0.01,
        jitter=1e-5,
    ):
        super().__init__()
        if d < 1:
            raise ValueError("d must be positive")
        if n_pseudo < 1:
            raise ValueError("n_pseudo must be positive")
        if scale <= 0 or lengthscale <= 0 or nugget <= 0 or jitter <= 0:
            raise ValueError("scale, lengthscale, nugget, and jitter must be positive")

        self.mean = MeanClass(*mean_class_init)
        self.d = int(d)
        self.n_pseudo = int(n_pseudo)
        self.jitter = float(jitter)

        lengthscale_t = torch.full(
            (self.d,), float(lengthscale), dtype=torch.get_default_dtype()
        )
        self.raw_scale = torch.nn.Parameter(
            torch.log(torch.as_tensor(scale, dtype=torch.get_default_dtype()))
        )
        self.raw_lengthscale = torch.nn.Parameter(torch.log(lengthscale_t))
        self.raw_nugget = torch.nn.Parameter(
            torch.log(torch.as_tensor(nugget, dtype=torch.get_default_dtype()))
        )

        # Repository inputs are normalized in the experiments, so [0, 1]^d is
        # a practical initialization. The locations remain unconstrained and
        # are optimized jointly with the covariance parameters.
        self.pseudo_inputs = torch.nn.Parameter(
            torch.rand(self.n_pseudo, self.d, dtype=torch.get_default_dtype())
        )

    @property
    def scale(self):
        return torch.exp(self.raw_scale)

    @property
    def lengthscale(self):
        return torch.exp(self.raw_lengthscale)

    @property
    def nugget(self):
        return torch.exp(self.raw_nugget)

    def _matern32(self, x1, x2):
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        distance = torch.cdist(x1_scaled, x2_scaled)
        sqrt3_distance = math.sqrt(3.0) * distance
        return (self.scale ** 2) * (1.0 + sqrt3_distance) * torch.exp(
            -sqrt3_distance
        )

    def covariance(self, x):
        """Return the batched SPGP/FITC covariance at ``x``."""
        squeeze_batch = x.dim() == 2
        x_view = x.unsqueeze(0) if squeeze_batch else x
        if x_view.dim() != 3 or x_view.size(-1) != self.d:
            raise ValueError(f"Expected x with shape [B, n, {self.d}] or [n, {self.d}]")

        pseudo = self.pseudo_inputs.to(device=x_view.device, dtype=x_view.dtype)
        kuu = self._matern32(pseudo, pseudo)
        eye_m = torch.eye(
            self.n_pseudo, device=x_view.device, dtype=x_view.dtype
        )
        chol_kuu = torch.linalg.cholesky(kuu + self.jitter * eye_m)

        kxu = self._matern32(x_view, pseudo)
        # A = K_uu^{-1} K_ux, batched over the leading dimensions.
        a = torch.cholesky_solve(kxu.transpose(-1, -2), chol_kuu)
        qxx = kxu @ a

        # A stationary Matérn parent kernel has constant marginal variance.
        correction = (self.scale ** 2 - torch.diagonal(qxx, dim1=-2, dim2=-1)).clamp_min(0.0)
        correction = correction + self.nugget
        cov = qxx + torch.diag_embed(correction)

        return cov.squeeze(0) if squeeze_batch else cov

    def cond_mean_sd(self, locs_batch, y_batch):
        """Compute the conditional predictive mean and standard deviation."""

        covmat = self.covariance(locs_batch)
        mean = self.mean(locs_batch)

        y_batch_demean = y_batch - mean

        chol = torch.linalg.cholesky(covmat)

        solved = torch.linalg.solve_triangular(
            chol,
            y_batch_demean,
            upper=False,
        )

        cond_mean = (
            chol[:, -1:, :-1]
            @ solved[:, :-1, :]
            + mean[:, -1:, :]
        )

        cond_sd = chol[:, -1:, -1:]

        return cond_mean, cond_sd

    def forward(self, locs_batch, y_batch):
        return self.cond_mean_sd(locs_batch, y_batch)
