import DeepKernelNNGP

def sim_kernel_config(d: int):
    kernel_config = {
        "MyMaternKernel": {"class": DeepKernelNNGP.MyMaternKernel, "init": [1.0, [0.3 for _ in range(d)], 1.5, 0.01]},
        "MyNSKernel_Scale": {"class": DeepKernelNNGP.MyNSKernel_Scale, "init": [0.0, 2.0, 2.0, 0.1, 0.5, 0.01]},
        "MyNSKernel_Lengthscale": {"class": DeepKernelNNGP.MyNSKernel_Lengthscale, "init": [-2.0, 1.0, -1.0, 1.0, 0.01]},
        "MyNSKernel_Kron": {"class": DeepKernelNNGP.MyNSKernel_Kron, "init": [0.3, 1.5, 0.01]},
        "LinearKernel": {"class": DeepKernelNNGP.LinearKernel, "init": [0.01]},
        "PeriodicKernel": {"class": DeepKernelNNGP.PeriodicKernel, "init": [1.0, 0.5, d, 0.01]},
        "TransformedMaternKernel": {"class": DeepKernelNNGP.TransformedMaternKernel, "init": [d, 1.0, 0.1 * (d ** 0.5), 1.5, 0.01]},
    }
    return kernel_config