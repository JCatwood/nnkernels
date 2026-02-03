import torch

class NllLoss(torch.nn.Module):
    """
    Negative log-likelihood based on univariate normal distributions
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, stderr=None, stderr_inv=None):
        assert pred.shape == target.shape
        constant = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        if stderr is not None:
            assert stderr.shape == target.shape
            nll = constant + torch.log(stderr) + 0.5 * (((pred - target) / stderr)**2)
        else:
            assert stderr_inv.shape == target.shape
            nll = constant - torch.log(stderr_inv) + 0.5 * (((pred - target) * stderr_inv)**2)
        return torch.mean(nll)

class MyMSELoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, pred, target, *args, **kwargs):
        pred_view = pred.squeeze()
        target_view = target.squeeze()
        assert pred_view.shape == target_view.shape
        return super().forward(pred_view, target_view)