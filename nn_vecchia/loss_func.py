import torch

def prepare_sequence_cond_sd(length, kernel, nbatch = 1):
    if isinstance(length, int):
        locs = torch.rand([nbatch, length, 2])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        return locs, L.to_dense()[:, -1, -1]
    else:
        assert len(length) == nbatch
        length_max = max(length)
        locs = torch.rand([nbatch, length_max, 2])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        mask = torch.arange(length_max).reshape(1, -1) < length.reshape(-1, 1)
        locs[~mask, :] = 0
        return locs, L.to_dense()[torch.arange(nbatch), length - 1, length - 1]

def prepare_sequence_cond_mean(length, kernel, nbatch = 1):
    if isinstance(length, int):
        locs = torch.rand([nbatch, length, 2])
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
        locs = torch.rand([nbatch, length_max, 2])
        covmat = kernel(locs)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (nbatch, length_max, 1))
        y = L @ x
        cond_mean = y[:, :, 0] - L.diagonal() * x[:, :, 0]
        cond_mean = cond_mean[torch.arange(nbatch), length - 1]
        mask = torch.arange(length_max).reshape(1, -1) < length.reshape(-1, 1)
        locs[~mask, :] = 0
        y[~mask, :] = 0
        y[torch.arange(nbatch), length - 1, :] = 0
        locs_and_y = torch.cat((locs, y), -1)
        return locs_and_y, cond_mean

def loss_len10_cond_sd(model, loss_function, kernel, nbatch):
    with torch.no_grad():
        locs, cond_sd = prepare_sequence_cond_sd(10, kernel, nbatch)
    cond_sd_pred, _, _ = model(locs, None, None)
    loss = loss_function(cond_sd_pred, cond_sd)
    return loss

def loss_len1to30_cond_sd(model, loss_function, kernel, nbatch):
    length = torch.randint(1, 30, (nbatch,))
    with torch.no_grad():
        locs, cond_sd = prepare_sequence_cond_sd(length, kernel, nbatch)
    cond_sd_pred, _, _ = model(locs, None, None)
    loss = loss_function(cond_sd_pred, cond_sd)
    return loss

def loss_len10_cond_mean(model, loss_function, kernel, nbatch):
    with torch.no_grad():
        locs_and_y, cond_mean = prepare_sequence_cond_mean(10, kernel, nbatch)
    cond_mean_pred, _, _ = model(locs_and_y, None, None)
    loss = loss_function(cond_mean_pred, cond_mean)
    return loss

def loss_len1to30_cond_mean(model, loss_function, kernel, nbatch):
    length = torch.randint(1, 30, (nbatch,))
    with torch.no_grad():
        locs_and_y, cond_mean = prepare_sequence_cond_mean(length, kernel, nbatch)
    cond_mean_pred, _, _ = model(locs_and_y, None, None)
    loss = loss_function(cond_mean_pred, cond_mean)
    return loss