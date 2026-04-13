import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Sequence, Tuple, Union


Tensor = torch.Tensor


class BaseVecchiaDataloader:
    """
    Base interface for Vecchia-style dataloaders.

    Subclasses should return batches in the form:

        X_batch : [B, m+1, d]
        y_batch : [B, m+1, 1]
        target  : task-dependent tensor

    where the final entry in `y_batch` is usually masked to zero and the
    corresponding value is returned in `target`.
    """

    def get_minibatch(self, *args, **kwargs):
        raise NotImplementedError

    def get_test_batch(self, *args, **kwargs):
        raise NotImplementedError


def _read_data(
    name: str,
    sep: str = ",",
    split: str = "train",
    floattype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Read feature and target tensors from:

        ./data/{name}/{split}/x.csv
        ./data/{name}/{split}/y.csv

    Parameters
    ----------
    name : str
        Dataset directory name.
    sep : str, default=","
        CSV separator.
    split : str, default="train"
        Dataset split, typically "train" or "test".
    floattype : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    X : torch.Tensor
        Feature matrix of shape [n, d].
    y : torch.Tensor
        Target matrix of shape [n, p].
    """
    file_path_X = f"./data/{name}/{split}/x.csv"
    file_path_y = f"./data/{name}/{split}/y.csv"

    X = torch.tensor(pd.read_csv(file_path_X, sep=sep, header=None).values, dtype=floattype)
    y = torch.tensor(pd.read_csv(file_path_y, sep=sep, header=None).values, dtype=floattype)
    return X, y


def _build_knn_indices(X_fit: Tensor, X_query: Tensor, n_neighbors: int) -> Tensor:
    """
    Build nearest-neighbor indices with sklearn.

    Parameters
    ----------
    X_fit : torch.Tensor
        Reference points of shape [n_fit, d].
    X_query : torch.Tensor
        Query points of shape [n_query, d].
    n_neighbors : int
        Number of neighbors to return.

    Returns
    -------
    idx : torch.Tensor
        Long tensor of shape [n_query, n_neighbors].
    """
    X_fit_np = X_fit.detach().cpu().numpy()
    X_query_np = X_query.detach().cpu().numpy()

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(X_fit_np)
    idx = nn.kneighbors(X_query_np, n_neighbors=n_neighbors, return_distance=False)
    return torch.from_numpy(idx).long()


class Vecc_Dataloader_GP_sim(torch.nn.Module, BaseVecchiaDataloader):
    """
    Synthetic Vecchia dataloader for Gaussian-process simulation experiments.

    This loader samples random input locations, generates a zero-mean GP draw
    using the provided covariance kernel, and returns Vecchia-style batches.

    Parameters
    ----------
    KernelCls : type
        Kernel class used to generate covariance matrices.
    kernel_parms : sequence
        Arguments passed to `KernelCls`.
    d : int, default=2
        Input dimension.
    target : str, default="y"
        Target type. One of:
        - "y"
        - "cond_mean"
        - "cond_sd"
        - "krig_coeff"
    device : torch.device or str, optional
        Device used for generated tensors.
    dtype : torch.dtype, default=torch.float32
        Floating-point dtype used for generated tensors.
    """

    def __init__(
        self,
        KernelCls,
        kernel_parms: Sequence,
        d: int = 2,
        target: Union[str, Tuple[str, ...]] = "y",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.kernel = KernelCls(*kernel_parms)
        self.d = d
        self.target = target[0] if isinstance(target, tuple) else target
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        assert self.target in ["y", "cond_mean", "cond_sd", "krig_coeff"], "invalid target input"

        self.kernel = self.kernel.to(self.device)

    def get_minibatch(
        self,
        size: int = 1024,
        m: int = 30,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate a synthetic Vecchia minibatch.

        Parameters
        ----------
        size : int, default=1024
            Batch size.
        m : int, default=30
            Conditioning-set size.

        Returns
        -------
        locs_batch : torch.Tensor
            Shape [size, m+1, d].
        y_batch : torch.Tensor
            Shape [size, m+1, 1], with final response masked to zero.
        target : torch.Tensor
            Target tensor for the selected task.
        """
        locs_batch = torch.rand(size, m + 1, self.d, device=self.device, dtype=self.dtype)
        covmat = self.kernel(locs_batch)
        L = torch.linalg.cholesky(covmat)

        x = torch.randn(size, m + 1, 1, device=self.device, dtype=self.dtype)
        y_batch = L @ x

        if self.target == "y":
            target = y_batch[:, -1:, :].clone()
        elif self.target == "cond_mean":
            target = y_batch[:, -1:, :] - L[:, -1:, -1:] * x[:, -1:, :]
        elif self.target == "cond_sd":
            target = L[:, -1:, -1:].clone()
        elif self.target == "krig_coeff":
            covmat_inv = torch.cholesky_inverse(L, upper=False)
            last_col = covmat_inv[:, :, -1:]
            target = -last_col[:, :-1, :] / last_col[:, -1:, :]
        else:
            raise RuntimeError("Unexpected target")

        y_batch = y_batch.clone()
        y_batch[:, -1:, :] = 0.0
        return locs_batch, y_batch, target

    def get_test_batch(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Alias for `get_minibatch`.
        """
        return self.get_minibatch(*args, **kwargs)


class Vecc_Dataloader_Dataset(BaseVecchiaDataloader):
    """
    Vecchia dataloader backed by train/test datasets stored on disk.

    Supports either a single dataset or multiple replicate datasets stored in
    subdirectories `seed_{seed}`.

    Parameters
    ----------
    data_name : str
        Dataset name under `./data/`.
    seeds : sequence, optional
        Replicate identifiers.
    floattype : torch.dtype, default=torch.float32
        Data dtype.
    """

    def __init__(
        self,
        data_name: str,
        seeds: Optional[Sequence[int]] = None,
        floattype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        self.multi_replicates = seeds is not None

        if self.multi_replicates:
            X_train_list = []
            y_train_list = []
            X_test_list = []
            y_test_list = []

            self.n_train = []
            self.n_test = []
            self.offset_train = [0]
            self.offset_test = [0]

            for seed in seeds:
                data_name_seed = f"{data_name}/seed_{seed}"

                X_seed, y_seed = _read_data(data_name_seed, split="train", floattype=floattype)
                X_train_list.append(X_seed)
                y_train_list.append(y_seed)
                self.n_train.append(X_seed.size(0))
                self.offset_train.append(self.offset_train[-1] + X_seed.size(0))

                X_seed, y_seed = _read_data(data_name_seed, split="test", floattype=floattype)
                X_test_list.append(X_seed)
                y_test_list.append(y_seed)
                self.n_test.append(X_seed.size(0))
                self.offset_test.append(self.offset_test[-1] + X_seed.size(0))

            self.X_train = torch.cat(X_train_list, dim=0)
            self.y_train = torch.cat(y_train_list, dim=0)
            self.X_test = torch.cat(X_test_list, dim=0)
            self.y_test = torch.cat(y_test_list, dim=0)
        else:
            self.X_train, self.y_train = _read_data(data_name, split="train", floattype=floattype)
            self.X_test, self.y_test = _read_data(data_name, split="test", floattype=floattype)

            self.n_train = [self.X_train.size(0)]
            self.n_test = [self.X_test.size(0)]
            self.offset_train = [0, self.X_train.size(0)]
            self.offset_test = [0, self.X_test.size(0)]

        self.d = self.X_train.size(1)
        self.n_replicates = len(self.n_train)
        self.NN_rev_train: Optional[Tensor] = None
        self.NN_test: Optional[Tensor] = None

    def update_NN(self, m: int = 30, scale: Optional[Tensor] = None) -> None:
        """
        Recompute nearest-neighbor indices.

        Parameters
        ----------
        m : int, default=30
            Conditioning-set size.
        scale : torch.Tensor, optional
            Feature scaling vector of shape [d]. Coordinates are divided by
            `scale` before nearest-neighbor search.
        """
        if scale is None:
            X_scaled_train = self.X_train
            X_scaled_test = self.X_test
        else:
            scale = scale.reshape(1, self.d)
            X_scaled_train = self.X_train / scale
            X_scaled_test = self.X_test / scale

        nn_rev_train_parts = []
        nn_test_parts = []

        for i in range(self.n_replicates):
            tr0, tr1 = self.offset_train[i], self.offset_train[i + 1]
            te0, te1 = self.offset_test[i], self.offset_test[i + 1]

            n_train_i = tr1 - tr0
            if n_train_i < m + 1:
                raise ValueError(
                    f"Replicate {i} has only {n_train_i} training points, but m+1={m+1} is required."
                )

            nn_train = _build_knn_indices(
                X_scaled_train[tr0:tr1, :],
                X_scaled_train[tr0:tr1, :],
                n_neighbors=m + 1,
            ) + tr0

            nn_rev_train_parts.append(nn_train[:, torch.arange(m, -1, -1)])

            nn_test = _build_knn_indices(
                X_scaled_train[tr0:tr1, :],
                X_scaled_test[te0:te1, :],
                n_neighbors=m,
            ) + tr0

            nn_test_parts.append(nn_test)

        self.NN_rev_train = torch.cat(nn_rev_train_parts, dim=0)
        self.NN_test = torch.cat(nn_test_parts, dim=0)

        assert torch.all(self.NN_rev_train[:, -1] == torch.arange(self.offset_train[-1]))

    def get_minibatch(
        self,
        size: int = 1024,
        m: int = 30,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample a Vecchia-style training minibatch.

        Returns
        -------
        X_batch : torch.Tensor
            Shape [B, m+1, d].
        y_batch : torch.Tensor
            Shape [B, m+1, 1], with target entry masked to zero.
        target : torch.Tensor
            Shape [B, 1, 1].
        """
        n_total = self.offset_train[-1]
        assert size <= n_total, f"size should be less than or equal to {n_total}"

        if self.NN_rev_train is None or self.NN_rev_train.size(1) < m + 1:
            self.update_NN(m=m)

        ind = torch.randperm(n_total)[:size]
        ind_NN = self.NN_rev_train[ind, -(m + 1):]

        X_batch = self.X_train[ind_NN, :]
        y_batch = self.y_train[ind_NN, :].clone()
        target = y_batch[:, -1:, :].clone()
        y_batch[:, -1:, :] = 0.0
        return X_batch, y_batch, target

    def get_test_batch(
        self,
        size: Union[int, str] = "all",
        m: int = 30,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build a Vecchia-style test batch.

        Parameters
        ----------
        size : int or "all", default="all"
            Number of test points to return.
        m : int, default=30
            Number of training neighbors.

        Returns
        -------
        X_batch : torch.Tensor
            Shape [B, m+1, d].
        y_batch : torch.Tensor
            Shape [B, m+1, 1], with the test response masked to zero.
        target : torch.Tensor
            Shape [B, 1, 1].
        """
        n_total = self.offset_test[-1]
        if size == "all":
            ind = torch.arange(n_total)
        else:
            assert isinstance(size, int)
            assert size <= n_total, f"size should be less than or equal to {n_total}"
            ind = torch.randperm(n_total)[:size]

        if self.NN_test is None or self.NN_test.size(1) < m:
            self.update_NN(m=m)

        ind_col = ind.unsqueeze(-1)
        X_test = self.X_test[ind_col, :]
        y_test = self.y_test[ind_col, :]

        ind = ind.view(-1)
        X_train = self.X_train[self.NN_test[ind, :m], :]
        y_train = self.y_train[self.NN_test[ind, :m], :].clone()

        zeros = torch.zeros(X_train.size(0), 1, 1, dtype=y_train.dtype, device=y_train.device)
        X_batch = torch.cat((X_train, X_test), dim=1)
        y_batch = torch.cat((y_train, zeros), dim=1)
        target = y_test
        return X_batch, y_batch, target