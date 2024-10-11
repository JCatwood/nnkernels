#!/bin/bash

source ./venv/bin/activate

for scene in Constant matern15 PiecewisePolynomialKernel RBFKernel exponential AdditiveKernel 
do
	nohup python3 spgp_test.py --scene=${scene} --N=20 --d=2 1>SPGP_${scene}.csv 2>SPGP_${scene}.err &
	nohup python3 nnspgp_test.py --scene=${scene} --N=100 --d=2 1>NNSPGP_${scene}.csv 2>NNSPGP_${scene}.err &
	nohup Rscript vgp_test.R ${scene} 100 2 1>VGP_${scene}.csv 2>VGP_${scene}.err &
done
