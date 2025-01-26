import torch
import torch.linalg as LA
from typing import Callable
from .criterion import DualCriterion
import copy

from math import log2

class TrustRegionOptimizer:
    """
    Solve the subproblem:
    J(x[B]) = f(x[B]) + 1/2 x[B]'@K[B,B]@x[B] + x[B]'@K[B,:]@x[!B], s.t. x[B] in [l, u]
    with turst region
    """
    
    def __init__(self, kkt_tol: float = 1.0e-6, cg_tol: float = 1.0e-8):
        self.kkt_tol = kkt_tol
        self.cg_tol = cg_tol

    def _quad_model(self, x: torch.Tensor, Q_linop: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor):
        """
        Input:
            x: (blk_size, n_target),
            Q: (n_target, n_target),
            g: (blk_size, n_target),
        Output:
            (n_target)
        """
        Qx = Q_linop(x)
        xQx = (x * Qx).sum(dim=0)
        gx = (g * x).sum(dim=0)
        return .5 * xQx + gx
    
    def _dual_objective(self, x: torch.Tensor, y: torch.Tensor, K_blk: torch.Tensor, K_grad_rest: torch.Tensor, criterion: DualCriterion):
        """
        Input:
            x: (blk_size, n_target),
            K_blk: (blk_size, blk_size),
            K_grad_rest: (blk_size, n_target),
            criterion: DualCriterion,
            y: (n_target)
        Output:
            (n_target)
        """
        Kgx = (x * K_grad_rest).sum(dim=0)
        xKx = (x * (K_blk @ x)).sum(dim=0)

        return Kgx + .5 * xKx + criterion(x, y)
    
    def check_optimality(self, alpha: torch.Tensor, grad: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, tol: float = 1.0e-6):
        """
        Check the optimality of the solution
        """
        pgrad = grad.clone()
        pgrad[alpha <= lower + tol].clamp_(min=0)
        pgrad[alpha >= upper - tol].clamp_(max=0)
        return pgrad.abs().max().item() < tol

    def minimize(self, criterion: DualCriterion, K_blk: torch.Tensor, K_grad_rest: torch.Tensor, alpha_init: torch.Tensor, y: torch.Tensor,
                    max_iter=50, eta=0.125, max_region_size=1.0, **kwargs):
        alpha = alpha_init
        region_size = max_region_size * 0.25 * torch.ones(y.size(1), device=y.device, dtype=y.dtype)
        
        # interface for preconditioning in the future
        precond = lambda x: x
        
        for lp_tr in range(max_iter):
            box_lower, box_upper = criterion.box(y)
            tr_lower = box_lower - alpha
            tr_upper = box_upper - alpha

            g = criterion.grad(alpha, y) + K_grad_rest + K_blk @ alpha

            # early stop trust region if KKT is satisfied
            if self.check_optimality(alpha, g, box_lower, box_upper):
                break

            Hd = criterion.hess(alpha, y)
            Q_linop: Callable[[torch.Tensor], torch.Tensor] = lambda x: Hd * x + K_blk @ x

            dual_obj = lambda x: self._dual_objective(x, y, K_blk, K_grad_rest, criterion)
            quad_mdl = lambda x: self._quad_model(x, Q_linop, g)

            # obtain step
            p = self._conjgrad_steihaug(Q_linop, g, tr_lower, tr_upper, region_size, precond=precond, **kwargs)
            quad_model_change = quad_mdl(p)

            # model change is too small, implying that there is almost no change in the objective.
            if quad_model_change.abs().max().item() < 1.0e-6:
                return alpha
            
            dual_obj_change = dual_obj(alpha+p) - dual_obj(alpha)
            
            # reduction ratio
            rho = dual_obj_change / quad_model_change
            # print(denominate, rho.item())
            if torch.isnan(rho).any():
                # nan means 0/0 occurs here. 
                # If 0 occurs at label not appearing then it is fine.
                unique_labels = torch.argmax(y, dim=1).unique()
                nan_indices = torch.isnan(rho)
                for idx in torch.where(nan_indices)[0]:
                    if idx not in unique_labels:
                        print(f"NaN detected at index {idx} with label not in unique labels {unique_labels}")
                        raise ValueError("rho is nan with labeled index")

            fail_step = (dual_obj_change >= 0) # fail step, the objective function ascent
            if fail_step.any():
                # shrink the region size for negative step
                region_size[fail_step] *= 0.25

            # check the update
            can_update = (rho > eta) & (~fail_step)
            if can_update.any():
                alpha[:,can_update] += p[:,can_update]

            # update the region size
            very_successful = (rho > 0.75) & ((p.norm(dim=0) - region_size).abs() < 1.0e-3) & (~fail_step)
            if very_successful.any():
                region_size[very_successful] = torch.clamp(2*region_size[very_successful], max=max_region_size)

            merely_successful = (rho < 0.25) & (~fail_step)
            if merely_successful.any():
                region_size[merely_successful] *= 0.5

            del very_successful, merely_successful, can_update, fail_step

        return alpha
    
    def _conjgrad_steihaug(self, B_linop: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor,
                                 l: torch.Tensor, u: torch.Tensor,
                                 region_size: torch.Tensor, cg_tol: float = 1e-8, cg_max_iter: int = 100,
                                 precond: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        p_return = torch.zeros_like(g)
        terminated = torch.zeros(g.size(1), dtype=torch.bool, device=g.device)

        p = torch.zeros_like(g)
        r = -g
        z = precond(r)
        d = z.clone() # this is correct version for preconditioning. TODO: test preconditioning

        # the tolerance is scaled by the number of targets
        cg_tol = self.cg_tol * g.size(1)

        r2old  = (r*z).sum(dim=0)
        r2init = r2old.clone()
        if r2old.sum() < cg_tol:
            return p
        
        for lp_cg in range(cg_max_iter):
            Bd = B_linop(d)
            dBd = (d * Bd).sum(dim=0)
            # skip the case when the curvature is negative since our problem ensures positive curvature
            # if dBd <= 0:
            #     raise ValueError("Negative curvature detected.")
            alpha = r2old / dBd
            p_next = p + alpha * d

            # check the non-terminated points that are out of the region
            out_region_ind = (torch.norm(p_next, dim=0) > region_size) & (~terminated)
            if out_region_ind.any():
                d_out = d[:, out_region_ind]
                p_out = p[:, out_region_ind]
                a = (d_out*d_out).sum(dim=0)
                b = 2 * (p_out*d_out).sum(dim=0)
                c = (p_out*p_out).sum(dim=0) - region_size[out_region_ind].pow(2)
                tau = (-b + torch.sqrt(b**2 - 4*a*c)) / (2*a)

                p_return[:, out_region_ind] = p_out + tau * d_out
                terminated[out_region_ind] = True

                # if all the points are on the boundary, then terminate the iteration
                if terminated.all().item():
                    break
            
            # check the non-terminated points that are out of the lower boundary or upper boundary
            out_bound_ind = ((p_next < l).any(dim=0) | (p_next > u).any(dim=0)) & (~terminated)
            if out_bound_ind.any().item():
                p_return[:, out_bound_ind] = p_next[:, out_bound_ind]
                terminated[out_bound_ind] = True
                # if all the points are on the boundary, then terminate the iteration
                if terminated.all():
                    break

            p = p_next.clone()
            r = r - alpha * Bd
            z = precond(r)
            r2new = (r * z).sum(dim=0)
            
            converged = (r2new < cg_tol * r2init) & (~terminated)
            if converged.any().item():
                p_return[:, converged] = p[:, converged]
                terminated[converged] = True
                # if all the points are on the boundary, then terminate the iteration
                if terminated.all():
                    break

            beta = r2new / r2old
            d = r + beta * d
            r2old = r2new.clone()

        p_return = torch.clamp(p_return, l, u)
        return p_return