import torch

class NllLoss(torch.nn.Module):
    """
    Negative log-likelihood based on univariate normal distributions
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, stderr):
        n = pred.size(0)
        constant = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        nll = constant + torch.log(stderr) + 0.5 * (((pred - target) / stderr)**2)
        return torch.mean(nll)