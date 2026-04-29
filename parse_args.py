import argparse
import DeepKernelNNGP
from kernel_config import sim_kernel_config


def parse_args():
    """
    Parse command-line arguments using argparse.
    """

    MEAN_MAP = {
        "NNMean": DeepKernelNNGP.NNMean,
        "ZeroMean": DeepKernelNNGP.ZeroMean,
        "ConstMean": DeepKernelNNGP.ConstMean,
    }

    valid_methods = ["VGP", "DeepKernelNNGP", "VGP_SM", "VGP_Wilson2015Deep"]

    parser = argparse.ArgumentParser(description="GP variants training script")

    parser.add_argument(
        "method",
        nargs="?",
        default="DeepKernelNNGP",
        choices=valid_methods,
        help="The model/methods to run",
    )

    parser.add_argument(
        "d",
        nargs="?",
        type=int,
        default=2,
        help="Number of features of the dataset",
    )

    parser.add_argument(
        "m",
        nargs="?",
        type=int,
        default=30,
        help="Size of the conditioning set",
    )

    parser.add_argument(
        "train_type",
        nargs="?",
        default="simulation",
        choices=["simulation", "data"],
        help="Training mode",
    )

    parser.add_argument(
        "wildcard_arg",
        nargs="?",
        default=None,
        help="kernel_gen_name (simulation) OR data_name (data)",
    )

    parser.add_argument(
        "n_replicates",
        nargs="?",
        type=int,
        default=None,
        help="Number of replicates (data mode only)",
    )

    args = parser.parse_args()

    kernel_config = sim_kernel_config(args.d)
    valid_kernels = list(kernel_config)

    if args.train_type == "simulation":
        kernel_gen_name = args.wildcard_arg or "MyNSKernel_Lengthscale"

        if kernel_gen_name not in kernel_config:
            raise ValueError(
                f"Invalid kernel_gen_name={kernel_gen_name!r}. "
                f"Expected one of {valid_kernels}."
            )

        KernelGen = kernel_config[kernel_gen_name]["class"]
        kernel_gen_init = kernel_config[kernel_gen_name]["init"]

        data_name = None
        n_replicates = None
        enforce_cross_group_nn = False
        group_ind_col = None
        max_nobs_per_group = None

    else:
        data_name = args.wildcard_arg or "GP_d2_rndlocs_mean0_Matern_2000_500"

        kernel_gen_name = None
        KernelGen = None
        kernel_gen_init = None

        n_replicates = args.n_replicates if args.n_replicates is not None else 1

        if data_name == "Argo":
            enforce_cross_group_nn = True
            group_ind_col = -1
            max_nobs_per_group = 200
        else:
            enforce_cross_group_nn = False
            group_ind_col = None
            max_nobs_per_group = None

    return {
        "method": args.method,
        "d": args.d,
        "m": args.m,
        "train_type": args.train_type,
        "kernel_gen_name": kernel_gen_name,
        "KernelGen": KernelGen,
        "kernel_gen_init": kernel_gen_init,
        "data_name": data_name,
        "n_replicates": n_replicates,
        "enforce_cross_group_nn": enforce_cross_group_nn,
        "group_ind_col": group_ind_col,
        "max_nobs_per_group": max_nobs_per_group,
    }