# Joker-paper
Implementation of "Joker: Joint Optimization Framework for Lightweight Kernel Machines".
The problem is:
$$
\min_{\theta}\frac12\langle\theta,\theta\rangle + c\sum_{i=1}^n\ell(y_i,\langle\theta,\varphi(x_i)\rangle),
$$
where $c$ is the penalty parameter, equivalent to $1/\lambda$ in the paper.

## How to use:
1. Download the dataset. Please follow `README.md` in each data fold.
2. Go to `src` folder, and run `python experiments.py`. Here are some examples:


```
# MSD
python experiments.py --dataset msd --criterion mse --c 1.0 --verbose_freq 5000 --max_iter 20000 --data_blksz 2048 --blksz 2048 --kernel rbf --n_rff -1 --max_trust_region_size 64 --region_shrink_rate 0.8
```

```
# HEPC
python experiments.py --dataset houseelec --criterion mse --c 128.0 --verbose_freq 5000 --max_iter 20000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000 --max_trust_region_size 64 --region_shrink_rate 0.6
```

```
# SUSY
python experiments.py --dataset houseelec --criterion svm --c 128.0 --verbose_freq 5000 --max_iter 20000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000 --max_trust_region_size 64 --region_shrink_rate 0.8
```

```
# HIGGS
python experiments.py --dataset higgs --criterion svm --c 128.0 --verbose_freq 5000 --max_iter 50000 --data_blksz 512 --blksz 512 --kernel lap --n_rff 100000 --max_trust_region_size 64 --region_shrink_rate 0.8
```

```
# CIFAR-5M
python experiments.py --dataset cifar5m --criterion svm --c 128.0 --verbose_freq 5000 --max_iter 20000 --data_blksz 512 --blksz 512 --kernel rbf --n_rff 200000 --max_trust_region_size 4 --region_shrink_rate 0.8
```
