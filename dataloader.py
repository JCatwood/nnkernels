import torch
import pandas
from sklearn.neighbors import NearestNeighbors

class Vecc_Dataloader_GP_sim(torch.nn.Module):
    def __init__(self, KernelCls, kernel_parms, d=2, target=('y', 'cond_mean', 'cond_sd', 'krig_coeff')):
        super().__init__()
        self.kernel = KernelCls(*kernel_parms)
        self.d = d
        if isinstance(target, tuple):
            self.target = target[0]
        else:
            self.target = target
        assert self.target in ['y', 'cond_mean', 'cond_sd', 'krig_coeff'], "invalid target input"
    
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
        elif self.target == "krig_coeff":
            covmat_inv = torch.cholesky_inverse(L, upper=False)
            invchol_lastcol = covmat_inv[:, :, -1:] / \
                (covmat_inv[:, -1:, -1:] ** 0.5) # [size, m+1, 1]
            target = - invchol_lastcol[:, :-1, :] / invchol_lastcol[:, -1:, :] # [size, m, 1]
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
    X = torch.tensor(pandas.read_csv(file_path_X, sep=sep, header=None).values,
                         dtype=floattype)
    y = torch.tensor(pandas.read_csv(file_path_y, sep=sep, header=None).values,
                         dtype=floattype)
    return X, y

class Vecc_Dataloader_Dataset:
    def __init__(self, data_name, seeds=None, *args, **kwargs):
        if "fixedlocs" in data_name:
            self.fixedlocs = True
        else:
            self.fixedlocs = False
        if seeds is not None:
            self.multi_replicates = True
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            self.n_train = []
            self.n_test = []
            self.offset_train = [0,]
            self.offset_test = [0,]
            for seed in seeds:
                data_name_seed = data_name + f"/seed_{seed}"
                X_seed, y_seed = _read_data(data_name_seed, type="train")
                X_train.append(X_seed)
                y_train.append(y_seed)
                self.n_train.append(X_seed.size(0))
                self.offset_train.append(self.offset_train[-1] + X_seed.size(0))
                X_seed, y_seed = _read_data(data_name_seed, type="test")
                X_test.append(X_seed)
                y_test.append(y_seed)
                self.n_test.append(X_seed.size(0))
                self.offset_test.append(self.offset_test[-1] + X_seed.size(0))
            self.X_train = torch.cat(X_train, dim=0)
            self.y_train = torch.cat(y_train, dim=0)
            self.X_test = torch.cat(X_test, dim=0)
            self.y_test = torch.cat(y_test, dim=0)
        else:
            self.multi_replicates = False
            self.X_train, self.y_train = _read_data(data_name, type="train")
            self.X_test, self.y_test = _read_data(data_name, type="test")
            self.n_train = [self.X_train.size(0),]
            self.n_test = [self.X_test.size(0),]
            self.offset_train = [0, self.X_train.size(0)]
            self.offset_test = [0, self.X_test.size(0)]
        self.d = self.X_train.size(1)
        self.n_replicates = len(self.n_train)
        self.NN_rev_train = None
        self.NN_test = None
    
    def update_NN(self, m=30, scale=None):
        if scale is None:
            X_scaled = self.X_train
        else:
            X_scaled = self.X_train * scale.reshape(1, self.d)
        self.NN_rev_train = []
        self.NN_test = []
        for i in range(self.n_replicates):
            NN_search_obj = NearestNeighbors(n_neighbors=m + 1, algorithm='auto')
            NN_search_obj.fit(X_scaled[self.offset_train[i] : self.offset_train[i + 1], :])
            NN_train = torch.from_numpy(NN_search_obj.kneighbors(
                X_scaled[self.offset_train[i] : self.offset_train[i + 1], :], 
                m + 1, return_distance=False)
                ) + self.offset_train[i]
            self.NN_rev_train.append(NN_train[:, torch.arange(m, -1, -1)])
            NN_test = torch.from_numpy(NN_search_obj.kneighbors(
                self.X_test[self.offset_test[i] : self.offset_test[i + 1], :],
                m, return_distance=False)
                ) + self.offset_train[i]
            self.NN_test.append(NN_test)
        self.NN_rev_train = torch.cat(self.NN_rev_train, dim=0)
        self.NN_test = torch.cat(self.NN_test, dim=0)
        assert torch.all(self.NN_rev_train[:, -1] == torch.arange(self.offset_train[-1]))

    def get_minibatch(self, size:int = 1024, m = 30, *args, **kwargs):
        assert size <= self.offset_train[-1], f"size should be less than or equal to {self.offset_train[-1]}"
        ind = torch.argsort(torch.rand(self.offset_train[-1]))[:size]
        if self.NN_rev_train is None or self.NN_rev_train.size(1) < m + 1:
            self.update_NN(m=m)
        ind_NN = self.NN_rev_train[ind, -(m + 1):]
        X_batch = self.X_train[ind_NN, :]
        y_batch = self.y_train[ind_NN, :]
        length = torch.full((size,), m + 1)
        target = y_batch[torch.arange(size), length - 1, 0].clone().unsqueeze(-1)
        y_batch[torch.arange(size), length - 1, 0] = 0.0
        return X_batch, y_batch, target, length
    
    def get_test_batch(self, size='all', m=30, *args, **kwargs):
        if size == 'all':
            ind = torch.arange(self.offset_test[-1])
        else:
            assert size <= self.offset_test[-1], f"size should be less than or equal to {self.offset_test[-1]}"
            ind = torch.argsort(torch.rand(self.offset_test[-1]))[:size]
        ind = ind.unsqueeze(-1) # [size, 1]
        if self.NN_test is None or self.NN_test.size(1) < m:
            self.update_NN_scale(m=m)
        X_test = self.X_test[ind, :] # [size, 1, d]
        y_test = self.y_test[ind, :] # [size, 1, 1]
        ind = ind.squeeze() # [size,]
        X_train = self.X_train[self.NN_test[ind, :m], :] # [size, m, d]
        y_train = self.y_train[self.NN_test[ind, :m], :] # [size, m, 1]
        X_batch = torch.cat((X_train, X_test), dim=1) # [size, m+1, d]
        y_batch = torch.cat((y_train, torch.zeros(X_batch.size(0), 1, 1)), dim=1) # [size, m+1, 1]
        length = torch.full((X_batch.size(0),), m+1)
        target = y_test.squeeze(-1) # [size, 1]
        return X_batch, y_batch, target, length