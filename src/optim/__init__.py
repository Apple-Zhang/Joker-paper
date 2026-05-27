from .trust_region import TrustRegionOptimizer
from .tncg import TNCGOptimizer


def make_optimizer(name: str = "trust_region", **kwargs):
    if name == "trust_region":
        return TrustRegionOptimizer(**kwargs)
    elif name == "tncg":
        return TNCGOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose 'trust_region' or 'tncg'.")
