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
def _read_data(name, sep=",", seed=None, type="train", floattype=torch.float32):
    if seed is None:
        file_path_X = f"./data/{name}/{type}/x.csv"
        file_path_y = f"./data/{name}/{type}/y.csv"
    else:
        file_path_X = f"./data/{name}/seed_{seed}/{type}/x.csv"
        file_path_y = f"./data/{name}/seed_{seed}/{type}/y.csv"

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
    def __init__(self, data_name, fixed_length=True, length_max=30, *args, **kwargs):
        self.X_train, self.y_train = _read_data(data_name, type="train", *args, **kwargs)
        self.X_test, self.y_test = _read_data(data_name, type="test", *args, **kwargs)
        self.fixed_length = fixed_length
        if fixed_length is False:
            raise ValueError("fixed_length = False is currently unsupported")
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
            print("Updating NN array...")
            self.update_NN_scale()
            print("Done")

        if ind is None:
            ind = torch.randperm(self.n_train)[:size]
        else:
            size = len(ind)
        X_batch = self.X_train[self.NN_train_rev[ind, :], :]
        y_batch = self.y_train[self.NN_train_rev[ind, :], :]
        length = torch.full((size,), self.length_max)
        target = y_batch[torch.arange(size), length - 1, 0].clone()
        y_batch[torch.arange(size), length - 1, 0] = 0.0
        return X_batch, y_batch, target, length
    
    def get_test_batch(self, ind=None, size:int = 1024, *args, **kwargs):
        if self.NN_train_rev is None or self.NN_test_rev is None:
            print("Updating NN array...")
            self.update_NN_scale()
            print("Done")
            
        if ind is None:
            if size > self.n_test:
                size = self.n_test
                print("In get_test_batch, the input size is bigger than n_test, using n_test instead")
            ind = torch.randperm(self.n_test)[:size]
        else:
            size = len(ind)
        X_batch = torch.cat((self.X_train[self.NN_test_rev[ind, :], :], 
                             self.X_test[ind, :].reshape(size, 1, self.d)), dim=1)
        y_batch = torch.cat((self.y_train[self.NN_test_rev[ind, :], :], 
                             torch.zeros(size, 1, 1)), dim=1)
        length = torch.full((size,), self.length_max)
        target = self.y_test[ind, 0].clone()
        return X_batch, y_batch, target, length