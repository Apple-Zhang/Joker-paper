import torch
import torch.nn.functional as F
from .fastfood import Fastfood

from typing import Callable

def _get_d2_matrix(x: torch.Tensor, y: torch.Tensor):
    # Compute the pairwise distance matrix
    d2x = (x*x).sum(dim=1)
    d2y = (y*y).sum(dim=1)
    d2 = d2x.unsqueeze_(1) + d2y.unsqueeze_(0)
    d2.addmm_(x, y.t(), beta=1, alpha=-2)
    d2.clamp_min_(0)
    return d2

def _gaussian(x: torch.Tensor, y: torch.Tensor, gamma: float):
    d2 = _get_d2_matrix(x, y)
    d2 *= -gamma
    d2.exp_()
    return d2

def _tstudent(x: torch.Tensor, y: torch.Tensor, gamma: float):
    d2 = _get_d2_matrix(x, y)
    d2 *= gamma
    d2.add_(1)
    d2.reciprocal_()
    return d2

class KernelBase:
    def __init__(self, kernel="rbf"):
        self.name = kernel
        self.apply_inexact_map = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Kernel not implemented")
    
    def init_inexact(self, dim: int, n_features: int, device="cuda:0", inexact_type="rff", dtype=torch.float32):
        raise NotImplementedError("Inexact model not implemented for this kernel yet")
    
    def __repr__(self) -> str:
        return f"Kernel({self.name})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name})"
    
class KeRBF(KernelBase):
    def __init__(self, gamma=1.0):
        super().__init__("rbf")
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return _gaussian(x, y, self.gamma)
    
    def init_inexact(self, dim: int, n_features: int, inexact_type="rff", device="cuda:0", dtype=torch.float32):
        if inexact_type == "rff":
            self.W = torch.randn(dim, n_features, device=device, dtype=dtype) * ((2.0 * self.gamma) ** 0.5)
            self.b = torch.rand(n_features, device=device, dtype=dtype) * (2 * 3.14159265358979323846)
            self.apply_inexact_map: Callable[[torch.Tensor], torch.Tensor] = \
                                    lambda x: torch.cos(torch.mm(x, self.W) + self.b) * ((2.0 / n_features) ** 0.5)
        elif inexact_type == "fastfood":
            sig = (1 / (2*self.gamma)) ** 0.5
            self.fastfood = Fastfood(dim, n_features // dim, sig, device=device, dtype=torch.float32)
            self.apply_inexact_map: Callable[[torch.Tensor], torch.Tensor] = self.fastfood.apply_feature_map
            print("Warning: Fastfood is experimental and may not work as expected")
        else:
            raise ValueError(f"Unsupported inexact type: {inexact_type}")
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"
    
    
class KeLinear(KernelBase):
    def __init__(self):
        super().__init__("linear")

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return x @ y.t()
    
    def init_inexact(self, dim: int, n_features: int, device="cuda:0", dtype=torch.float64):
        self.apply_inexact_map: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    
    def __repr__(self) -> str:
        return f"Kernel({self.name})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name})"
    
class KePoly(KernelBase):
    def __init__(self, degree=2):
        super().__init__("poly")
        self.degree = degree

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return (x @ y.t() + 1) ** self.degree
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, degree={self.degree})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name}, degree={self.degree})"
    
class KeSigmoid(KernelBase):
    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__("sigmoid")
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return torch.tanh(self.alpha * x @ y.t() + self.beta)
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, alpha={self.alpha}, beta={self.beta})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name}, alpha={self.alpha}, beta={self.beta})"
    
class KeStudenT(KernelBase):
    def __init__(self, gamma=1.0):
        super().__init__("Student-T")
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return _tstudent(x, y, self.gamma)
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"
    
class KeLaplace(KernelBase):
    def __init__(self, gamma=1.0):
        super().__init__("lap")
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        gram = torch.cdist(x, y, p=1)
        gram *= -self.gamma
        gram.exp_()
        return gram
        # return torch.exp(-dist_matrix * self.gamma)
    
    def init_inexact(self, dim: int, n_features: int, device="cuda:0", inexact_type="rff", dtype=torch.float64):
        if inexact_type == "rff":
            cauchy = torch.distributions.cauchy.Cauchy(0, self.gamma)
            W = cauchy.sample((dim, n_features)).to(device=device, dtype=dtype)
            b = torch.rand(n_features, device=device, dtype=dtype) * (2 * 3.14159265358979323846)
            self.apply_inexact_map: Callable[[torch.Tensor], torch.Tensor] = \
                                    lambda x: torch.cos(torch.mm(x, W) + b) * ((2.0 / n_features) ** 0.5)
        elif inexact_type == "fastfood":
            raise NotImplementedError("Fastfood not implemented for Laplace kernel")
        else:
            raise ValueError(f"Unsupported inexact type: {inexact_type}")
    
    def __repr__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"
    
    def __str__(self) -> str:
        return f"Kernel({self.name}, gamma={self.gamma})"

def make_kernel(kernel: str, **kwargs):
    if kernel == "rbf":
        return KeRBF(kwargs["gamma"])
    elif kernel == "lap":
        return KeLaplace(kwargs["gamma"])
    elif kernel == "linear":
        return KeLinear()
    elif kernel == "poly":
        return KePoly(kwargs["d"])
    elif kernel == "sigmoid":
        return KeSigmoid(kwargs["gamma"])
    elif kernel == "student":
        return KeStudenT(kwargs["gamma"])
    else:
        raise ValueError(f"Unknown kernel: {kernel}")