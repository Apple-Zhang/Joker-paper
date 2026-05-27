from .collections import *
    
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
        return EpsInsensitive(c=c, eps=kwargs["eps"])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
__all__ = ["make_criterion", "MeanSquareError", "Huber", "SquaredHinge", "LogLoss", "EpsInsensitive"]