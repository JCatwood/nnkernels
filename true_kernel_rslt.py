import torch
import json

from models import GPVecchia, MyMaternKernel, MyNSKernel_Scale, MyNSKernel_Lengthscale
from dataloader import Vecc_Dataloader_Dataset
from loss import NllLoss

torch.manual_seed(1)
# %% tuning parameters
d = 2  # locs are sampled from R^d, only used when train_type is "simulation"
m = 30
data_names = ["GP_d2_rndlocs_mean0_Matern_2000_500", 
             "GP_d2_rndlocs_mean0_NS_range_2000_500", 
             "GP_d2_rndlocs_mean0_NS_scale_2000_500"]  # only used when train_type is "data"
kernel_classes = [MyMaternKernel, MyNSKernel_Lengthscale, MyNSKernel_Scale]
kernel_init_parms = [[1.0, 0.03, 1.5, 0.01], 
                     [-4., 4., -4., 1.0, 0.01], 
                     [0.0, 15., 20., 0.03, 0.5, 0.01]]
n_replicates = 100
for data_name, kernel_class, kernel_init_parm in zip(data_names, kernel_classes, kernel_init_parms):
    data_seeds = range(n_replicates)
    dataloader = Vecc_Dataloader_Dataset(data_name, data_seeds)
    model = GPVecchia(kernel_class, *kernel_init_parm)
    model.eval()
    loss_MSE = torch.nn.MSELoss()
    loss_NLL = NllLoss()
    with torch.no_grad():
        X_batch, y_batch, y_true, length = dataloader.get_test_batch(size='all', m=m)
        y_pred, y_stderr = model(X_batch, y_batch, length=length)
        loss_NLL_val = loss_NLL(y_pred, y_true, y_stderr)
        loss_MSE_val = loss_MSE(y_pred, y_true)
        print(">>>")
        output_dict = {
            "data_type": "data",
            "data_name": data_name,
            "model": "GPVecchia",
            "kernel_train": "truth",
            "m": m,
            "NLL": loss_NLL_val.item(),
            "MSE": loss_MSE_val.item(),
            "n_replicates": n_replicates,
        }
        output_str = json.dumps(output_dict)
        print(output_str)
        print("<<<")
