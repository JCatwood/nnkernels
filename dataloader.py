import torch
import pandas
from sklearn.neighbors import NearestNeighbors

class Vecc_Dataloader_GP_sim(torch.nn.Module):
    def __init__(self, KernelCls, kernel_parms, d=2, fixed_length=False, length_max=30, 
                 target=('y', 'cond_mean', 'cond_sd', 'inv_chol')):
        super().__init__()
        self.kernel = KernelCls(*kernel_parms)
        self.d = d
        self.fixed_length = fixed_length
        self.length_max = length_max
        if isinstance(target, tuple):
            self.target = target[0]
        else:
            self.target = target
        assert self.target in ['y', 'cond_mean', 'cond_sd', 'inv_chol'], "invalid target input"
    
    def get_minibatch(self, ind=None, size:int = 1024, m:int = 30, *args, **kwargs):
        if m is None:
            length_max = self.length_max
        else:
            length_max = m + 1
        locs_batch = torch.rand(size, length_max, self.d)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)
        x = torch.normal(0.0, 1.0, (size, length_max, 1))
        y_batch = L @ x
        if not self.fixed_length:
            length = torch.randint(1, length_max, (size, ))
            mask = torch.arange(length_max).reshape(1, -1) < length.reshape(-1, 1)
            locs_batch[~mask, :] = 0
            y_batch[~mask, :] = 0
        else:
            length = torch.full((size,), length_max)
        if self.target == 'y':
            target = y_batch[torch.arange(size), length - 1, 0].clone().unsqueeze(-1)
        elif self.target == 'cond_mean':
            target = (y_batch[torch.arange(size), length - 1, 0] - \
                L[torch.arange(size), length - 1, length - 1] * x[torch.arange(size), length - 1, 0]).unsqueeze(-1)
        elif self.target == 'cond_sd':
            target = L[torch.arange(size), length - 1, length - 1].unsqueeze(-1)
        elif self.target == "inv_chol":
            assert self.fixed_length == True, "does not support different lengths when the target is inv_chol"
            covmat_inv = torch.cholesky_inverse(L, upper=False)
            target = covmat_inv[:, :, -1:] / \
                (covmat_inv[:, -1:, -1:] ** 0.5)
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
        self.N = len(self.offset_train)
        # self.NN_train_rev = None
        # self.NN_test_rev = None
        self.n_train = int(self.X_train.size(0) / self.N)
        self.n_test = int(self.X_test.size(0) / self.N)
        self.d = self.X_train.size(1)
    
    # not use NN for training
    # def update_NN_scale(self, scale=None):
    #     if scale is None:
    #         X_scaled = self.X_train
    #         X_test_scaled = self.X_test
    #     else:
    #         X_scaled = self.X_train * scale.reshape(1, self.d)
    #         X_test_scaled = self.X_test * scale.reshape(1, self.d)
    #     NN_search_obj = NearestNeighbors(n_neighbors=self.default_len - 1, algorithm='auto')
    #     NN_search_obj.fit(X_scaled)
    #     NN_train = torch.from_numpy(NN_search_obj.kneighbors(X_scaled, self.default_len, return_distance=False))
    #     NN_test = torch.from_numpy(NN_search_obj.kneighbors(X_test_scaled, self.default_len - 1, return_distance=False))
    #     self.NN_train_rev = NN_train[:, torch.arange(self.default_len - 1, -1, -1)]
    #     self.NN_test_rev = NN_test[:, torch.arange(self.default_len - 2, -1, -1)]
    #     assert torch.all(self.NN_train_rev[:, -1] == torch.arange(self.n_train))

    def get_minibatch(self, ind=None, size:int = 1024, m = 30, n_replicates='all', *args, **kwargs):
        if n_replicates == 'all':
            n_replicates = self.N
        else:
            assert isinstance(n_replicates, int) and n_replicates > 0, "n_replicates should be a positive integer or 'all'"
            assert n_replicates <= self.N, f"n_replicates should be less than or equal to {self.N}"
        rnd_tmp = torch.rand((size, self.n_train))
        ind = rnd_tmp.argsort(dim=-1)[:, :m+1]
        seed_ind = torch.randint(0, n_replicates, (size, 1))
        ind = ind + seed_ind * self.n_train
        X_batch = self.X_train[ind, :]
        y_batch = self.y_train[ind, :]
        length = torch.full((size,), m + 1)
        target = y_batch[torch.arange(size), length - 1, 0].clone().unsqueeze(-1)
        y_batch[torch.arange(size), length - 1, 0] = 0.0
        return X_batch, y_batch, target, length
    
    def get_test_batch(self, seed=0, m=30, use_NN=False, *args, **kwargs):
        X_train = self.X_train[self.offset_train[seed] : self.offset_train[seed] + self.n_train]
        X_test = self.X_test[self.offset_test[seed] : self.offset_test[seed] + self.n_test]
        y_train = self.y_train[self.offset_train[seed] : self.offset_train[seed] + self.n_train]
        y_test = self.y_test[self.offset_test[seed] : self.offset_test[seed] + self.n_test]
        if use_NN:
            NN_search_obj = NearestNeighbors(n_neighbors=m+1, algorithm='auto')
            NN_search_obj.fit(X_train)
            NN_test = torch.from_numpy(NN_search_obj.kneighbors(X_test, m, return_distance=False))
            X_batch = torch.cat((X_train[NN_test, :], X_test.unsqueeze(1)), dim=1)
            y_batch = torch.cat((y_train[NN_test, :], torch.zeros(X_batch.size(0), 1, 1)), dim=1)
        else:
            rnd_tmp = torch.rand((X_test.size(0), self.n_train))
            ind = rnd_tmp.argsort(dim=-1)[:, :m]
            X_batch = torch.cat((X_train[ind, :], X_test.unsqueeze(1)), dim=1)
            y_batch = torch.cat((y_train[ind, :], torch.zeros(X_batch.size(0), 1, 1)), dim=1)
        
        length = torch.full((X_batch.size(0),), m+1)
        target = y_test.clone()
        return X_batch, y_batch, target, length