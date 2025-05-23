# Joker-paper
Implementation of "Joker: Joint Optimization Framework for Lightweight Kernel Machines".
The problem is:
```math
\min_{\theta}\frac12\langle\theta,\theta\rangle + c\sum_{i=1}^n\ell(y_i,\langle\theta,\varphi(x_i)\rangle),
```
where $c$ is the penalty parameter, equivalent to $1/\lambda$ in the paper.

## How to use:
1. Download the dataset. Please follow `README.md` in each data folder.
2. Go to `src` folder, and run `python experiments.py`. Here are some examples:


```
# MSD
python experiments.py --dataset msd --criterion mse --c 1.0 --verbose_freq 5000 --max_iter 10000 --data_blksz 2048 --blksz 2048 --kernel rbf --n_rff -1
```

```
# HEPC
python experiments.py --dataset houseelec --criterion mse --c 128.0 --verbose_freq 5000 --max_iter 20000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000
```

```
# SUSY
python experiments.py --dataset susy --criterion svm --c 32.0 --verbose_freq 5000 --max_iter 15000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000
```

```
# HIGG
python experiments.py --dataset higgs --criterion svm --c 128.0 --verbose_freq 5000 --max_iter 35000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000
```

```
# CIFAR-5M
python experiments.py --dataset cifar5m --criterion svm --c 128.0 --verbose_freq 5000 --max_iter 40000 --data_blksz 512 --blksz 512 --kernel rbf --n_rff 200000
```
