from typing import Callable, Optional
import torch
from criterion.collections import DualCriterion


def _truncated_conjgrad(A_lin_op: Callable[[torch.Tensor], torch.Tensor],
                       b: torch.Tensor,
                       x0: Optional[torch.Tensor] = None,
                       l: Optional[torch.Tensor] = None,
                       u: Optional[torch.Tensor] = None,
                       rel_tol: float = 0.5, max_iter: int = 100,
                       precond: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> torch.Tensor:
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_lin_op(x)
    z = precond(r)
    rz_old = (r * z).sum(dim=0)
    p = z.clone()

    eta = min(rel_tol, b.abs().sum().item() ** 0.5)

    for i in range(max_iter):
        Ap = A_lin_op(p)
        pAp = (p * Ap).sum(dim=0)

        valid = pAp > 0
        if not valid.any():
            break

        alpha_cg = torch.zeros_like(rz_old)
        alpha_cg[valid] = rz_old[valid] / pAp[valid]

        x_next = x + alpha_cg * p

        if l is not None and u is not None:
            x_next.clamp_(l, u)

        x = x_next
        r = b - A_lin_op(x)

        if r.norm() < eta:
            break

        z = precond(r)
        rz_new = (r * z).sum(dim=0)

        beta = torch.zeros_like(rz_old)
        beta[valid] = rz_new[valid] / rz_old[valid]
        p = z + beta * p
        rz_old = rz_new.clone()

    if l is not None and u is not None:
        x.clamp_(l, u)

    return x


class TNCGOptimizer:
    def __init__(self, kkt_tol: float = 1.0e-6, cg_tol: float = 1.0e-8):
        self.kkt_tol = kkt_tol
        self.cg_tol = cg_tol

    def check_optimality(self, alpha: torch.Tensor, grad: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, tol: float = 1.0e-6):
        pgrad = grad.clone()
        pgrad[alpha <= lower + tol].clamp_(min=0)
        pgrad[alpha >= upper - tol].clamp_(max=0)
        return pgrad.abs().max().item() < tol

    def minimize(self, criterion: DualCriterion,
                       K_blk: torch.Tensor,
                       K_grad_rest: torch.Tensor,
                       alpha_init: torch.Tensor,
                       y: torch.Tensor,
                       max_iter=50, **kwargs):

        alpha = alpha_init
        box_lower, box_upper = criterion.box(y)

        for lp in range(max_iter):
            tr_lower = box_lower - alpha
            tr_upper = box_upper - alpha

            g = criterion.grad(alpha, y) + K_grad_rest + K_blk @ alpha

            if self.check_optimality(alpha, g, box_lower, box_upper):
                break

            Hd = criterion.hess(alpha, y)
            Q_linop: Callable[[torch.Tensor], torch.Tensor] = lambda x: Hd * x + K_blk @ x

            # Newton direction via projected CG
            s = _truncated_conjgrad(A_lin_op=Q_linop, b=-g, l=tr_lower, u=tr_upper, max_iter=10)

            gTs = (g * s).sum()
            if gTs < 0:
                sHs = (s * Q_linop(s)).sum()
                t = (-gTs / sHs).clamp(max=1.0) if sHs > 0 else 1.0
            else:
                # CG failed — fall back to projected gradient step
                pgrad = g.clone()
                pgrad[alpha <= box_lower + self.kkt_tol].clamp_(min=0)
                pgrad[alpha >= box_upper - self.kkt_tol].clamp_(max=0)
                s = -pgrad
                sHs = (s * Q_linop(s)).sum()
                gTs = (g * s).sum()
                t = (-gTs / sHs).clamp(max=1.0) if sHs > 0 and gTs < 0 else 0.01

            alpha += t * s
            alpha.clamp_(box_lower, box_upper)

        return alpha
