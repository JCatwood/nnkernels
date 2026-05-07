import torch

def input_transformed_dim(d, type="locs_diff"):
    if type == "locs":
        return d
    elif type == "locs_diff":
        return d
    elif type == "dist_direction":
        return d + 1
    elif type == "dist_direction_lastloc":
        return 2 * d + 1
    elif type == "locs_lastloc":
        return 2 * d
    else:
        raise ValueError("Unexpected type for input_transform")
    
def input_transform(locs_batch, type="locs_diff"):
    """
    Process the batched versions of locs and y
    :param locs_batch: locations, n_batch X mplus1 X d
    :param type: a string for the type of transformation
    """
    n_batch, mplus1, d = locs_batch.shape
    if type == "locs":
        return locs_batch
    if type == "locs_diff":
        locs_batch_trans = locs_batch - locs_batch[:, -1:, :]
        return locs_batch_trans
    elif type == "dist_direction":
        locs_batch_trans = locs_batch - locs_batch[:, -1:, :]
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist.clamp_min(1e-12)
        direction[:, -1:, :] = 0
        return torch.cat((dist, direction), dim=-1)
    elif type == "dist_direction_lastloc":
        lastloc_batch = locs_batch[:, -1:, :]
        locs_batch_trans = locs_batch - lastloc_batch
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist.clamp_min(1e-12)
        direction[:, -1:, :] = 0
        lastloc_batch_vew = lastloc_batch.expand(n_batch, mplus1, d)
        return torch.cat((dist, direction, lastloc_batch_vew), dim=-1)
    elif type == "locs_lastloc":
        lastloc_batch = locs_batch[:, -1:, :]
        lastloc_batch_vew = lastloc_batch.expand(n_batch, mplus1, d)
        return torch.cat((locs_batch, lastloc_batch_vew), dim=-1)
    else:
        raise ValueError("Unexpected type for input_transform")

