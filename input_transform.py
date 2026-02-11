import torch

def input_transformed_dim(d, type="locs_diff"):
    if type == "locs":
        return d
    elif type == "locs_and_y":
        return d + 1
    elif type == "locs_diff":
        return d
    elif type == "locs_diff_and_y":
        return d + 1
    elif type == "dist_direction":
        return d + 1
    elif type == "dist_direction_and_y":
        return d + 2
    elif type == "dist_direction_lastloc":
        return 2 * d + 1
    elif type == "dist_direction_lastloc_and_y":
        return 2 * d + 2
    elif type == "locs_lastloc":
        return 2 * d
    elif type == "locs_lastloc_and_y":
        return 2 * d + 1
    
def input_transform(locs_batch, y_batch=None, length=None, type="locs_diff"):
    """
    Process the batched versions of locs and y
    :param locs_batch: locations, n_batch X length_max X d
    :param y: responses, n_batch X length_max X 1
    :param length: None or a single int or a tensor of int
    :param type: a string for the type of transformation
    """
    n_batch, length_max, d = locs_batch.shape
    if isinstance(length, int) or length is None:
            length = torch.full((n_batch,), length_max, dtype=torch.tensor(1).dtype)
    if type == "locs":
        return locs_batch
    elif type == "locs_and_y":
        y_batch_trans = y_batch.clone()
        y_batch_trans[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((locs_batch, y_batch_trans), dim=-1)
    if type == "locs_diff":
        locs_batch_trans = locs_batch - locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        return locs_batch_trans
    elif type == "locs_diff_and_y":
        locs_batch_trans = locs_batch - locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        y_batch_trans = y_batch.clone()
        y_batch_trans[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((locs_batch_trans, y_batch_trans), dim=-1)
    elif type == "dist_direction":
        locs_batch_trans = locs_batch - locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist
        direction[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((dist, direction), dim=-1)
    elif type == "dist_direction_and_y":
        locs_batch_trans = locs_batch - locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist
        direction[torch.arange(n_batch), length - 1, :] = 0
        y_batch_trans = y_batch.clone()
        y_batch_trans[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((dist, direction, y_batch_trans), dim=-1)
    elif type == "dist_direction_lastloc":
        lastloc_batch = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        locs_batch_trans = locs_batch - lastloc_batch
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist
        direction[torch.arange(n_batch), length - 1, :] = 0
        lastloc_batch_vew = lastloc_batch.expand(n_batch, length_max, d)
        return torch.cat((dist, direction, lastloc_batch_vew), dim=-1)
    elif type == "dist_direction_lastloc_and_y":
        lastloc_batch = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        locs_batch_trans = locs_batch - lastloc_batch
        dist = torch.linalg.norm(locs_batch_trans, dim=-1, keepdim=True)
        direction = locs_batch_trans / dist
        direction[torch.arange(n_batch), length - 1, :] = 0
        lastloc_batch_vew = lastloc_batch.expand(n_batch, length_max, d)
        y_batch_trans = y_batch.clone()
        y_batch_trans[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((dist, direction, lastloc_batch_vew, y_batch_trans), dim=-1)
    elif type == "locs_lastloc":
        lastloc_batch = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        lastloc_batch_vew = lastloc_batch.expand(n_batch, length_max, d)
        return torch.cat((locs_batch, lastloc_batch_vew), dim=-1)
    elif type == "locs_lastloc_and_y":
        lastloc_batch = locs_batch[torch.arange(n_batch), length - 1, :].unsqueeze(1)
        lastloc_batch_vew = lastloc_batch.expand(n_batch, length_max, d)
        y_batch_trans = y_batch.clone()
        y_batch_trans[torch.arange(n_batch), length - 1, :] = 0
        return torch.cat((locs_batch, lastloc_batch_vew, y_batch_trans), dim=-1)


