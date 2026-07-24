# About
This repository provides code to reproduce the results in the paper [Permutation-preserving Functions and Neural Vecchia Covariance Kernels](URL).

# Directory
- `pkg`: the developed python package implementing the proposed Neural Vecchia (**NeuVec**) method as well as the classic Vecchia approximation of GPs.
- `argo_data.py`: download and pre-process the Argo dataset
- `kernel_config.py`: configurations for kernels used for data simulation in the simulation study
- `models.py`: initialization of the models to be trained (i.e., NeuVec and three benchmarks)
- `parse_args.py`: script for parsing command line arguments used in `main.py`
- `true_kernel_rslt.py`: script for computing the MSE and NLL using the true kernel in the simulation study
- `main.py`: the main script for simulation and real-data study
- `results.txt`: the consolidated results from running `main.py` with different arguments
- `process_sim_rslt.R`: the script for producing the illustrations in the paper using data from `results.txt`

# To Reproduce
Here we don't specify the python/R versions but using the latest stable versions are recommended. These are the commands I used in my Mac Terminal.
```
python3 -m venv venv
source venv/bin/activate
pip install -e pkg
```
Also need to install other dependent modules with
```
pip install pandas argopy 
```

## Simulation 

To run the four methods over four simulation scenarios using different `m` (conditioning set size), create the following bash script, for example, named `sim.sh`.
```
#!/bin/bash
set -euo pipefail

source venv/bin/activate

d=3

for kernel_gen_name in MyMaternKernel MyNSKernel_Lengthscale PeriodicKernel TransformedMaternKernel; do
    for method in VGP VGP_SM VGP_Wilson2015Deep DeepKernelNNGP SPGP; do
        for m in 10 30 50 70 90; do

            echo "======================================"
            echo "Running: method=$method, kernel=$kernel_gen_name, m=$m"
            echo "======================================"

            python3 main.py "$method" "$d" "$m" simulation "$kernel_gen_name" \
                > "${method}_${kernel_gen_name}_${d}_${m}_sim.out" 2>&1

        done
    done
done
```
Run the above script with 
```
chmod +x sim.sh
./sim.sh
```
through command line.

## Results for using the true kernels
Simply run
```
python3 true_kernel_rslt.py 3 30
```
where `3` is `d` and `30` is `m`.

## Argo data
First, download and process the Argo data by
```
python3 argo_data.py
```
Then create the following bash script, for example, named `data.sh`.
```
#!/bin/bash
set -euo pipefail

source venv/bin/activate

d=4
m=30

for data_name in Argo; do
    for method in VGP VGP_SM VGP_Wilson2015Deep DeepKernelNNGP SPGP; do

        echo "======================================"
        echo "Running: method=$method, data=$data_name"
        echo "======================================"

        python3 main.py "$method" "$d" "$m" data "$data_name" 17

    done
done
```
Here, `17` is the number of years, also the number of replicates. Run the above script with 
```
chmod +x data.sh
./data.sh
```

## Reproduce illustrations
Extract all lines between `>>>` and `<<<` in the `.out` output files into the `results.txt` file. Then run 
```
Rscript process_sim_rslt.R
```
PS: you may need to install the required R packages from CRAN.


