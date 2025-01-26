# Description: Fastfood kernel approximation.
import torch
import torch.distributions
import numpy as np
from scipy.special import gammaincinv

gpu_hadamard_available = True
try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    # print("Warning: fast_hadamard_transform not found, Fastfood not available.")
    gpu_hadamard_available = False
    from scipy.linalg import hadamard

# implementation refers to: https://github.com/jizhaox/fastfood-kernel/blob/master/FastfoodForKernel.m
class Fastfood:
    def __init__(self, input_dim: int, dim_multiplier: int, sigma: float, transform_type="fft", device="cpu", dtype=torch.float64):
        self.input_dim = input_dim
        self.output_dim = input_dim * dim_multiplier
        self.dim_multiplier = dim_multiplier
        self.sigma = sigma
        self.dtype = dtype
        self.device = torch.device(device)

        if self.device.type == "cpu" or gpu_hadamard_available == False:
            self._fwht = self._fwht_cpu
        else:
            self._fwht = self._fwht_gpu

        self.B = (torch.rand(self.output_dim, device=device, dtype=dtype) * 2 - 1).sign()
        self.P = [torch.randperm(input_dim, device=device, dtype=torch.int64) for i in range(dim_multiplier)]
        self.G = torch.randn(self.output_dim, device=device, dtype=dtype)

        # chi distribution
        # http://en.wikipedia.org/wiki/Chi_distribution
        # sampling via cumulative distribution function
        chi = gammaincinv(input_dim / 2, np.random.rand(self.output_dim))
        chi = np.sqrt(chi * 2)
        for i in range(self.dim_multiplier):
            start_ind = self.input_dim * (i)
            end_ind   = self.input_dim * (i+1)
            chi[start_ind:end_ind] = chi[start_ind:end_ind] / np.linalg.norm(self.G[start_ind:end_ind])
        self.S = torch.from_numpy(chi).to(device=device, dtype=dtype)

    # https://github.com/Dao-AILab/fast-hadamard-transform
    def _fwht_gpu(self, x: torch.Tensor) -> torch.Tensor:
        return hadamard_transform(x, scale=1.0)
    
    def _fwht_cpu(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FWHT for CPU is not implemented")
    
    def do_fastfood(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(x.size(0), self.output_dim, device=x.device, dtype=torch.float32)
        x = x.to(torch.float32)
        for i in range(self.dim_multiplier):
            start_ind = self.input_dim * (i)
            end_ind   = self.input_dim * (i+1)
            z = x.view(-1, self.input_dim)
            z = z * self.B[start_ind:end_ind]
            z = self._fwht(z)
            z = z[self.P[i]]
            z = z * self.G[start_ind:end_ind]
            z = self._fwht(z)
            z = z * self.S[start_ind:end_ind]
            res[start_ind:end_ind] = z
        res = res.to(self.dtype)
        return res / (self.sigma * (self.output_dim ** 0.5))
    
    def apply_feature_map(self, x: torch.Tensor):
        x = self.do_fastfood(x)
        b = torch.rand(self.output_dim, device=x.device, dtype=x.dtype) * (2 * 3.14159265358979323846)
        return torch.cos(x + b) * (2.0 / np.sqrt(self.output_dim))