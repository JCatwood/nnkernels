import torch
import pandas
from sklearn.neighbors import NearestNeighbors

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

class Vecc_Dataloader_GP_sim:
    def __init__(self, kernel, d=2, fixed_length=False, length_max=30, target=('y', 'cond_mean', 'cond_sd')):
        self.kernel = kernel
        self.d = d
        self.fixed_length = fixed_length
        self.length_max = length_max
        if isinstance(target, tuple):
            self.target = target[0]
        else:
            self.target = target
        assert self.target in ['y', 'cond_mean', 'cond_sd'], "invalid target input"
    
    def get_minibatch(self, ind=None, size:int = 1024, *args, **kwargs):
        locs_batch = torch.rand(size, self.length_max, self.d)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (size, self.length_max, 1))
        y_batch = L @ x
        if not self.fixed_length:
            length = torch.randint(1, self.length_max, (size, ))
            mask = torch.arange(self.length_max).reshape(1, -1) < length.reshape(-1, 1)
            locs_batch[~mask, :] = 0
            y_batch[~mask, :] = 0
        else:
            length = torch.full((size,), self.length_max)
        if self.target == 'y':
            target = y_batch[torch.arange(size), length - 1, 0].clone()
        elif self.target == 'cond_mean':
            target = y_batch[torch.arange(size), length - 1, 0] - \
                L[torch.arange(size), length - 1, length - 1] * x[torch.arange(size), length - 1, 0]
        elif self.target == 'cond_sd':
            target = L[torch.arange(size), length - 1, length - 1]
        else:
            raise Exception("Unexpected self.target")
        y_batch[torch.arange(size), length - 1, :] = 0
        return locs_batch, y_batch, target, length
    
    def get_test_batch(self, *args, **kwargs):
        return self.get_minibatch(*args, **kwargs)


"""Read data (train, valid, test) from the local data directory

Parameters
----------
name : str
    directory name, for example, "mtcars", "gp_matern15"
sep : char, optional
    character use to separate entries
type: str, optional
    indicate whether to read from the "train", "valid", or "test" sub-folder

Returns
-------
tuple
    a tuple of X and y, features and labels
"""
def _read_data(name, sep=",", type="train", floattype=torch.float32):
    file_path_X = f"./data/{name}/{type}/X.csv"
    file_path_y = f"./data/{name}/{type}/y.csv"

    try:
        X = torch.tensor(pandas.read_csv(file_path_X, sep=sep, header=None).values,
                         dtype=floattype)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_X}")
        return None
    except pandas.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path_X}")
        return None
    except pandas.errors.ParserError:
         print(f"Error: Failed to parse CSV file: {file_path_X}. "
               "Check the delimiter and file format.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    try:
        y = torch.tensor(pandas.read_csv(file_path_y, sep=sep, header=None).values,
                         dtype=floattype)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_y}")
        return None
    except pandas.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path_y}")
        return None
    except pandas.errors.ParserError:
         print(f"Error: Failed to parse CSV file: {file_path_y}. "
               "Check the delimiter and file format.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    return X, y 

class Vecc_Dataloader_Dataset:
    def __init__(self, data_name, fixed_length=False, length_max=30, *args, **kwargs):
        self.X_train, self.y_train = _read_data(data_name, type="train", *args, **kwargs)
        self.X_test, self.y_test = _read_data(data_name, type="test", *args, **kwargs)
        self.fixed_length = fixed_length
        self.length_max = length_max
        self.NN_train_rev = None
        self.NN_test_rev = None
        self.n_train = self.X_train.size(0)
        self.n_test = self.X_test.size(0)
        self.d = self.X_train.size(1)
    
    def update_NN_scale(self, scale=None):
        if scale is None:
            X_scaled = self.X_train
            X_test_scaled = self.X_test
        else:
            X_scaled = self.X_train * scale.reshape(1, self.d)
            X_test_scaled = self.X_test * scale.reshape(1, self.d)
        NN_search_obj = NearestNeighbors(n_neighbors=self.length_max - 1, algorithm='auto')
        NN_search_obj.fit(X_scaled)
        NN_train = torch.from_numpy(NN_search_obj.kneighbors(X_scaled, self.length_max, return_distance=False))
        NN_test = torch.from_numpy(NN_search_obj.kneighbors(X_test_scaled, self.length_max - 1, return_distance=False))
        self.NN_train_rev = NN_train[:, torch.arange(self.length_max - 1, -1, -1)]
        self.NN_test_rev = NN_test[:, torch.arange(self.length_max - 2, -1, -1)]
        assert torch.all(self.NN_train_rev[:, -1] == torch.arange(self.n_train))

    def get_minibatch(self, ind=None, size:int = 1024, *args, **kwargs):
        if self.NN_train_rev is None or self.NN_test_rev is None:
            self.update_NN_scale()

        if ind is None:
            ind = torch.randperm(self.n_train)[:size]
        else:
            size = len(ind)
        X_batch = self.X_train[self.NN_train_rev[ind, :], :]
        y_batch = self.y_train[self.NN_train_rev[ind, :], :]
        if not self.fixed_length:
            length = torch.randint(1, self.length_max, (size, ))
            mask = torch.arange(self.length_max).reshape(1, -1) < length.reshape(-1, 1)
            X_batch[~mask, :] = 0
            y_batch[~mask, :] = 0
        else:
            length = torch.full((size,), self.length_max)
        target = y_batch[torch.arange(size), length - 1, 0].clone()
        y_batch[torch.arange(size), length - 1, 0] = 0.0
        return X_batch, y_batch, target, length
    
    def get_test_batch(self, ind=None, size:int = 1024, *args, **kwargs):
        if self.NN_train_rev is None or self.NN_test_rev is None:
            self.update_NN_scale()
            
        if ind is None:
            ind = torch.randperm(self.n_test)[:size]
        else:
            size = len(ind)
        X_batch = torch.cat((self.X_train[self.NN_test_rev[ind, :], :], self.X_test[ind, :].reshape(size, 1, self.d)), dim=1)
        y_batch = torch.cat((self.y_train[self.NN_test_rev[ind, :], :], torch.zeros(size, 1, 1)), dim=1)
        length = torch.full((size,), self.length_max)
        target = self.y_test[ind, 0].clone()
        return X_batch, y_batch, target, length