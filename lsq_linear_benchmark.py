import torch
import time
import sys
import json

from models import (
    NNDT_Sum_NNTG,
    MyMaternKernel,
    MyNSKernel_Scale,
    MyNSKernel_Lengthscale,
)
from loss import NllLoss, MyMSELoss
from dataloader import Vecc_Dataloader_GP_sim, Vecc_Dataloader_Dataset
from input_transform import input_transformed_dim, input_transform

torch.manual_seed(1)
# %% tuning parameters
# To run the least-sq benchmark, the dataset needs to be N replicates of the same locations, with N1 replicates being the training dataset and N2 being testing. N1 + N2 = N. 
d = 2  # locs are sampled from R^d
m = 30
use_NN_for_testing = True
use_NN_for_training = True 
n_replicates_for_training = 'all'  # can be 'all' or a positive integer 
cond_on_train = False 
train_type = "data"
if len(sys.argv) > 1:
    data_name = sys.argv[1]
else:
    data_name = f"GP_d{d}_fixedlocs_mean0_NS_range_80_20"
# %% dataloader
dataloader = Vecc_Dataloader_Dataset(data_name)
# %% scheduler
def lr_lambda(epoch):
    base_lr = 0.001
    factor = 0.0001
    return base_lr / (1 + factor * epoch)
    # return 0.001
# %% evaluate LSQ benchmark
loss_MSE = torch.nn.MSELoss()
with torch.no_grad():
    lsq = torch.zeros(dataloader.n_test, m)
    for i in range(dataloader.n_test):
        X_batch_train, y_batch_train, y_true_train, length = dataloader.get_minibatch(
            ind=i + torch.arange(dataloader.N_train)*dataloader.n_train, m=m, 
            n_replicates='all', use_NN=use_NN_for_testing
        )
        y_batch_train = y_batch_train[:, :-1, :]
        lsq_sol = torch.linalg.lstsq(y_batch_train.squeeze(-1), y_true_train)
        lsq[i, :] = lsq_sol.solution.squeeze(-1)
    
    MSE_lst = torch.zeros(dataloader.N_test)
    for i in range(dataloader.N_test):
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(
            seed=i, m=m, use_NN=use_NN_for_testing, cond_on_train=cond_on_train
        )
        y_batch = y_batch[:, :-1, :]
        y_pred_lsq = torch.sum(lsq.unsqueeze(-1) * y_batch, dim=1)
        MSE_lst[i] = loss_MSE(y_pred_lsq, y_true).item()
    output_dict = {
        "data_type": train_type,
        "data_name": data_name,
        "model": "LSQ",
        "m": m,
        "MSE": MSE_lst.mean().item(),
    }
    output_str = json.dumps(output_dict)
    print(output_str)
        

# %%
