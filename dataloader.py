import torch
import pandas
from sklearn.neighbors import NearestNeighbors

class Vecc_Dataloader_GP_sim(torch.nn.Module):
    def __init__(self, KernelCls, kernel_parms, d=2, target=('y', 'cond_mean', 'cond_sd')):
        super().__init__()
        self.kernel = KernelCls(*kernel_parms)
        self.d = d
        if isinstance(target, tuple):
            self.target = target[0]
        else:
            self.target = target
        assert self.target in ['y', 'cond_mean', 'cond_sd'], "invalid target input"
    
    def get_minibatch(self, ind=None, size:int = 1024, m:int = 30, *args, **kwargs):
        locs_batch = torch.rand(size, m+1, self.d)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (size, m+1, 1))
        y_batch = L @ x
        length = torch.full((size,), m+1)
        if self.target == 'y':
            target = y_batch[torch.arange(size), length - 1, 0].clone().unsqueeze(-1)
        elif self.target == 'cond_mean':
            target = (y_batch[torch.arange(size), length - 1, 0] - \
                L[torch.arange(size), length - 1, length - 1] * x[torch.arange(size), length - 1, 0]).unsqueeze(-1)
        elif self.target == 'cond_sd':
            target = L[torch.arange(size), length - 1, length - 1].unsqueeze(-1)
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
    file_path_X = f"./data/{name}/{type}/x.csv"
    file_path_y = f"./data/{name}/{type}/y.csv"
    file_path_offset = f"./data/{name}/{type}_offset.csv"
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
    
    try:
        offset = torch.tensor(pandas.read_csv(file_path_offset, sep=sep, header=None).values,
                         dtype=int)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_offset}")
        return None
    except pandas.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path_offset}")
        return None
    except pandas.errors.ParserError:
         print(f"Error: Failed to parse CSV file: {file_path_offset}. "
               "Check the delimiter and file format.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    return X, y, offset

class Vecc_Dataloader_Dataset:
    def __init__(self, data_name, *args, **kwargs):
        if "fixedlocs" in data_name:
            self.fixedlocs = True
        else:
            self.fixedlocs = False
        self.X_train, self.y_train, self.offset_train = _read_data(data_name, type="train", *args, **kwargs)
        self.X_test, self.y_test, self.offset_test = _read_data(data_name, type="test", *args, **kwargs)
        self.offset_train = self.offset_train.squeeze()
        self.offset_test = self.offset_test.squeeze()
        self.N_train = len(self.offset_train)
        self.N_test = len(self.offset_test)
        self.NN_train_rev = None
        self.n_train = int(self.X_train.size(0) / self.N_train)
        self.n_test = int(self.X_test.size(0) / self.N_test)
        self.d = self.X_train.size(1)
    
    def update_NN_scale(self, m=30, scale=None):
        if scale is None:
            X_scaled = self.X_train[:self.n_train, :]
        else:
            X_scaled = self.X_train[:self.n_train, :] * scale.reshape(1, self.d)
        NN_search_obj = NearestNeighbors(n_neighbors=m + 1, algorithm='auto')
        NN_search_obj.fit(X_scaled)
        NN_train = torch.from_numpy(NN_search_obj.kneighbors(X_scaled, m + 1, return_distance=False))
        self.NN_train_rev = NN_train[:, torch.arange(m, -1, -1)]
        assert torch.all(self.NN_train_rev[:, -1] == torch.arange(self.n_train))

    def get_minibatch(self, ind=None, size:int = 1024, m = 30, n_replicates='all', 
                      use_NN=False, *args, **kwargs):
        if n_replicates == 'all':
            n_replicates = self.N_train
        else:
            assert isinstance(n_replicates, int) and n_replicates > 0, "n_replicates should be a positive integer or 'all'"
            assert n_replicates <= self.N_train, f"n_replicates should be less than or equal to {self.N_train}"
        if use_NN:
            if not self.fixedlocs:
                raise ValueError("use NN for training is not supported for datasets with different locations between replicates")
            if ind is None:
                ind = torch.argsort(torch.rand(n_replicates * self.n_train))[:size]
            else:
                size = len(ind)
            ind_target_loc = ind % self.n_train
            ind_replicate = ind // self.n_train
            if self.NN_train_rev is None or self.NN_train_rev.size(1) < m + 1:
                    self.update_NN_scale(m=m)
            ind_NN = self.NN_train_rev[ind_target_loc, -(m + 1):] + \
                ind_replicate.unsqueeze(-1) * self.n_train
        else:
            if ind is not None:
                Warning("ind is ignored when use_NN is False")
            rnd_tmp = torch.rand((size, self.n_train))
            ind_replicate = torch.randint(0, n_replicates, (size, 1))
            ind_NN = rnd_tmp.argsort(dim=-1)[:, :m+1] + ind_replicate * self.n_train
        X_batch = self.X_train[ind_NN, :]
        y_batch = self.y_train[ind_NN, :]
        length = torch.full((size,), m + 1)
        target = y_batch[torch.arange(size), length - 1, 0].clone().unsqueeze(-1)
        y_batch[torch.arange(size), length - 1, 0] = 0.0
        return X_batch, y_batch, target, length
    
    def get_test_batch(self, seed=0, m=30, use_NN=False, cond_on_train=True, *args, **kwargs):
        assert seed < self.N_test, f"seed should be less than {self.N_test}"
        X_test = self.X_test[self.offset_test[seed] : self.offset_test[seed] + self.n_test]
        y_test = self.y_test[self.offset_test[seed] : self.offset_test[seed] + self.n_test]
        if cond_on_train:
            assert self.N_train == self.N_test, "cond_on_train=True is only supported when the number of training replicates is the same as the number of test replicates"
            X_train = self.X_train[self.offset_train[seed] : self.offset_train[seed] + self.n_train]
            y_train = self.y_train[self.offset_train[seed] : self.offset_train[seed] + self.n_train]
        if use_NN and cond_on_train:
            NN_search_obj = NearestNeighbors(n_neighbors=m, algorithm='auto')
            NN_search_obj.fit(X_train)
            NN_test = torch.from_numpy(NN_search_obj.kneighbors(X_test, m, return_distance=False))
            NN_test_rev = NN_test[:, torch.arange(m-1, -1, -1)]
            X_batch = torch.cat((X_train[NN_test_rev, :], X_test.unsqueeze(1)), dim=1)
            y_batch = torch.cat((y_train[NN_test_rev, :], torch.zeros(X_batch.size(0), 1, 1)), dim=1)
        elif use_NN and not cond_on_train:
            NN_search_obj = NearestNeighbors(n_neighbors=m+1, algorithm='auto')
            NN_search_obj.fit(X_test)
            NN_test = torch.from_numpy(NN_search_obj.kneighbors(X_test, m+1, return_distance=False))
            NN_test_rev = NN_test[:, torch.arange(m, -1, -1)]
            assert torch.all(NN_test_rev[:, -1] == torch.arange(self.n_test))
            X_batch = X_test[NN_test_rev, :]
            y_batch = y_test[NN_test_rev, :]
            y_batch[:, -1, :] = 0.0
        elif not use_NN and cond_on_train:
            rnd_tmp = torch.rand((X_test.size(0), self.n_train))
            ind = rnd_tmp.argsort(dim=-1)[:, :m]
            X_batch = torch.cat((X_train[ind, :], X_test.unsqueeze(1)), dim=1)
            y_batch = torch.cat((y_train[ind, :], torch.zeros(X_batch.size(0), 1, 1)), dim=1)
        else:
            raise ValueError("cond_on_train=False and use_NN=False is not supported for test batch")
        
        length = torch.full((X_batch.size(0),), m+1)
        target = y_test.clone()
        return X_batch, y_batch, target, length