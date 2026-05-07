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

class NNMean(torch.nn.Module):
    def __init__(self, d, sizes, dropout=0.0):
        assert isinstance(sizes, (list, tuple)), "sizes must be a list or tuple"
        super().__init__()
        sizes_all = [d,] + list(sizes) + [1,]
        self.NN = _build_block(sizes_all, dropout=dropout)
    
    def forward(self, X):
        """
        Input Shapes
        ------------
        X : Tensor
            Input tensor of shape `[N, d]` or `[B, N, d]`.

        Returns
        -------
        Tensor
            Mean values of shape `[N, 1]` or `[B, N, 1]`.
        """
        return self.NN(X)

class ZeroMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        """
        Input Shapes
        ------------
        X : Tensor
            Input tensor of shape `[N, d]` or `[B, N, d]`.

        Returns
        -------
        Tensor
            Zero tensor of shape `[N, 1]` or `[B, N, 1]`.
        """
        return X.new_zeros(*X.shape[:-1], 1)


class ConstMean(torch.nn.Module):
    def __init__(self, const_mean):
        super().__init__()
        self.const_mean = torch.nn.Parameter(
            torch.as_tensor(const_mean, dtype=torch.get_default_dtype())
        )
    
    def forward(self, X):
        """
        Input Shapes
        ------------
        X : Tensor
            Input tensor of shape `[N, d]` or `[B, N, d]`.

        Returns
        -------
        Tensor
            Constant tensor of shape `[N, 1]` or `[B, N, 1]`.
        """
        return X.new_ones(*X.shape[:-1], 1) * self.const_mean
