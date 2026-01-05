import torch

def prepare_seq(locs, y, NN_rev, size=None, index=None):
    """
    Prepare mini-batch style location and response-y arrays
    
    :param locs: locations, n X d
    :param y: responses, length n
    :param NN_rev: reverse conditioning sets, n X (m + 1), last column is 0:n-1
    :param size: size of mini-batch, must be provided if index is None
    :param index: indices of the mini-batch
    """

    assert locs.size(0) == y.size(0)
    assert y.size(0) == NN_rev.size(0)
    n = locs.size(0)
    d = locs.size(1)
    m = NN_rev.size(1) - 1
    if index is not None:
        pass
    elif size is None:
        raise Exception("size and index cannot be both None")
    else:
        index = torch.randperm(n)[:size]
    locs_batch = locs[NN_rev[index, :], :]
    y_batch = y[NN_rev[index, :]]
    return locs_batch, y_batch
    