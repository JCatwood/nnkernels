import torch

class NllLoss(torch.nn.Module):
    """
    Negative log-likelihood based on univariate normal distributions
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, stderr):
        pred_view = pred.squeeze()
        target_view = target.squeeze()
        stderr_view = stderr.squeeze()
        assert pred_view.shape == target_view.shape
        assert stderr_view.shape == target_view.shape
        n = pred.size(0)
        constant = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        nll = constant + torch.log(stderr) + 0.5 * (((pred - target) / stderr)**2)
        return torch.mean(nll)

class MyMSELoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, pred, target, *args, **kwargs):
        pred_view = pred.squeeze()
        target_view = target.squeeze()
        assert pred_view.shape == target_view.shape
        return super().forward(pred_view, target_view)