import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from math import exp, log

class DualCriterion(ABC):
    def __init__(self, c: float = 1.0):
        self.c = c # c = 1 / reg_lambda
        self.task_type = None

    @abstractmethod
    def __call__(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def grad(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def hess(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def hess(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def box(self, y: torch.Tensor) -> torch.Tensor:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c})"
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c})"
    
def make_criterion(criterion: str, c: float = 1.0, **kwargs) -> DualCriterion:
    if criterion == "mse":
        return MeanSquareError(c=c)
    elif criterion == "huber":
        return Huber(c=c, delta=kwargs["delta"])
    elif criterion == "svm":
        return SquaredHinge(c=c)
    elif criterion == "log":
        return LogLoss(c=c, dtype=kwargs["dtype"])
    elif criterion == "svr":
        return eps_insensitive(c=c, eps=kwargs["eps"])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
        
class LogLoss(DualCriterion):
    def __init__(self, c: float = 1.0, dtype: torch.dtype = torch.float64): # c = 1 / reg_lambda
        super(LogLoss, self).__init__(c)
        self.task_type = "classification"

        self.c_eps = torch.nextafter(torch.tensor(c, dtype=dtype), torch.tensor(0.0, dtype=dtype)).item()
        self.eps = c - self.c_eps
        self.box_lower = self.eps
        self.box_upper = self.c_eps

        self.max_hess = (1/self.eps) ** 0.5
        self.const = -self.c * log(self.c) # the extra constant term in the dual objective, never affect the solution

    def __call__(self, alpha: torch.Tensor, y: torch.Tensor):
        alpha_y = alpha * y
        return (alpha_y * alpha_y.log()).sum(dim=0) + ((self.c-alpha_y) * (self.c-alpha_y).log()).sum(dim=0) - self.const * y.size(0)

    def grad(self, alpha: torch.Tensor, y: torch.Tensor):
        alpha_y = alpha * y
        grad = y * (alpha_y/(self.c-alpha_y)).log()
        # close_to_lower = (alpha_y < self.theta)
        # close_to_upper = ((self.c - alpha_y) < self.theta)
        # grad[close_to_lower] = self.sur_curve_grad * y[close_to_lower]
        # grad[close_to_upper] = -(self.sur_curve_grad + self.sur_curve_hess * self.c) * y[close_to_upper]

        return grad

    def hess(self, alpha: torch.Tensor, y: torch.Tensor):
        alpha_y = alpha * y
        hess = self.c / (alpha_y * (self.c - alpha_y))

        # to avoid cancellation, hess should have a upper bound
        hess = torch.clamp(hess, max=self.max_hess)

        return hess
    
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor):
        return F.softplus(-y_pred * y).sum() * self.c
    
    def box(self, y: torch.Tensor):
        mask_pos = y > 0
        box_lower = self.box_lower * torch.ones_like(y)
        box_upper = self.box_upper * torch.ones_like(y)
        box_lower[~mask_pos] = -self.box_upper
        box_upper[~mask_pos] = -self.box_lower
        return box_lower, box_upper
    
    def __str__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c}, eps={self.eps}, max_hess={self.max_hess})"
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c}, eps={self.eps}, max_hess={self.max_hess})"
    
class SquaredHinge(DualCriterion):
    def __init__(self, c: float = 1.0):
        super(SquaredHinge, self).__init__(c)
        self.task_type = "classification"
        self.box_lower = 0
        self.box_upper = float("inf")

    def __call__(self, alpha: torch.Tensor, y: torch.Tensor): # c = 1 / reg_lambda
        return 0.5 * (alpha*alpha).sum(dim=0) / self.c - (alpha*y).sum(dim=0)

    def grad(self, alpha: torch.Tensor, y: torch.Tensor):
        return alpha / self.c - y

    def hess(self, alpha: torch.Tensor, y: torch.Tensor):
        return 1/self.c * torch.ones_like(alpha)
    
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor):
        return 0.5 * F.relu(1 - y_pred * y).pow(2).sum() * self.c
    
    def box(self, y: torch.Tensor):
        box_lower = -torch.inf * torch.ones_like(y)
        box_upper =  torch.inf * torch.ones_like(y)
        mask_pos = y > 0
        box_lower[mask_pos] = 0
        box_upper[~mask_pos] = 0
        return box_lower, box_upper

class MeanSquareError(DualCriterion):
    def __init__(self, c: float = 1.0):
        super(MeanSquareError, self).__init__(c)
        self.task_type = "regression"
        self.box_lower = -float("inf")
        self.box_upper = float("inf")

    def __call__(self, alpha: torch.Tensor, y: torch.Tensor): # c = 1 / reg_lambda
        return 0.5 * (alpha*alpha).sum(dim=0) / self.c - (alpha*y).sum(dim=0)

    def grad(self, alpha: torch.Tensor, y: torch.Tensor):
        return alpha / self.c - y

    def hess(self, alpha: torch.Tensor, y: torch.Tensor):
        return 1/self.c * torch.ones_like(alpha)
    
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor):
        return 0.5 * (y_pred - y).pow(2).sum() * self.c
    
    def box(self, y: torch.Tensor):
        return self.box_lower * torch.ones_like(y), self.box_upper * torch.ones_like(y)
    
class Huber(DualCriterion):
    def __init__(self, c: float = 1.0, delta: float = 0.5):
        super(Huber, self).__init__(c)
        self.task_type = "regression"
        self.box_lower = -c*delta
        self.box_upper = c*delta
        self.delta = delta

    def __call__(self, alpha: torch.Tensor, y: torch.Tensor): # c = n / reg_lambda
        return 0.5 * (alpha*alpha).sum(dim=0) / self.c - (alpha*y).sum(dim=0)

    def grad(self, alpha: torch.Tensor, y: torch.Tensor):
        return alpha / self.c - y

    def hess(self, alpha: torch.Tensor, y: torch.Tensor):
        return 1/self.c * torch.ones_like(alpha)
    
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor):
        return F.huber_loss(y_pred, y, delta=self.delta, reduction='sum') * self.c
    
    def box(self, y: torch.Tensor):
        return self.box_lower * torch.ones_like(y), self.box_upper * torch.ones_like(y)
    
    def __str__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c}, delta={self.delta})"
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(c={self.c}, delta={self.delta})"
    
class eps_insensitive(DualCriterion):
    def __init__(self, c: float = 1.0, eps: float = 0.1):
        super(eps_insensitive, self).__init__(c)
        self.task_type = "regression"
        self.box_lower = -c
        self.box_upper = c
        self.eps = eps

    def __call__(self, alpha: torch.Tensor, y: torch.Tensor):
        return alpha.abs().sum(dim=0) * self.eps - (alpha*y).sum(dim=0)
    
    def grad(self, alpha: torch.Tensor, y: torch.Tensor):
        return alpha.sign() * self.eps - y
    
    def hess(self, alpha: torch.Tensor, y: torch.Tensor):
        return torch.zeros_like(alpha)
   
    def primal_obj(self, y_pred: torch.Tensor, y: torch.Tensor):
        loss = ((y_pred - y).abs() - self.eps).clamp(min=0)
        return loss.sum() * self.c
    
    def box(self, y: torch.Tensor):
        return self.box_lower * torch.ones_like(y), self.box_upper * torch.ones_like(y)