import torch

def prepare_sequence_cond_sd(length, kernel, nbatch=1, d=2):
    if isinstance(length, int):
        locs = torch.rand([nbatch, length, d])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        return locs, L.to_dense()[:, -1, -1]
    else:
        assert len(length) == nbatch
        length_max = max(length)
        locs = torch.rand([nbatch, length_max, d])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        mask = torch.arange(length_max).reshape(1, -1) < length.reshape(-1, 1)
        locs[~mask, :] = 0
        return locs, L.to_dense()[torch.arange(nbatch), length - 1, length - 1]

def prepare_sequence_cond_mean(length, kernel, nbatch=1, d=2):
    if isinstance(length, int):
        locs = torch.rand([nbatch, length, d])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (nbatch, length, 1))
        y = L @ x
        cond_mean = (L[:, (length - 1):, :(length - 1)] @ x[:, :(length - 1), :]).squeeze()
        y[:, -1, :] = 0
        locs_and_y = torch.cat((locs, y), -1)
        return locs_and_y, cond_mean
    else:
        assert len(length) == nbatch
        length_max = max(length)
        locs = torch.rand([nbatch, length_max, d])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (nbatch, length_max, 1))
        y = L @ x
        cond_mean = y[:, :, 0] - \
            L[:, torch.arange(length_max), torch.arange(length_max)] * x[:, :, 0]
        cond_mean = cond_mean[torch.arange(nbatch), length - 1]
        mask = torch.arange(length_max).reshape(1, -1) < length.reshape(-1, 1)
        locs[~mask, :] = 0
        y[~mask, :] = 0
        y[torch.arange(nbatch), length - 1, :] = 0
        locs_and_y = torch.cat((locs, y), -1)
        return locs_and_y, cond_mean
    
def prepare_sequence_locs_and_y(length, kernel, nbatch=1, d=2):
    assert isinstance(length, int)
    locs = torch.rand([nbatch, length, d])
    covmat = kernel(locs)
    L = torch.linalg.cholesky(covmat)
    x = torch.normal(0.0, 1.0, (nbatch, length, 1))
    y = (L @ x).squeeze(-1)
    return locs, y
    

def sim_GP(nTrain, nTest, kernel, d=2):
    n = nTrain + nTest
    locs = torch.rand([n, d])
    covmat = kernel(locs)
    L = torch.linalg.cholesky(covmat)
    x = torch.normal(0.0, 1.0, (n, 1))
    y = (L @ x).squeeze()
    locs_train = locs[:nTrain, :]
    locs_test = locs[nTrain:, :]
    y_train = y[:nTrain]
    y_test = y[nTrain:]
    return locs_train, locs_test, y_train, y_test