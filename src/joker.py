import torch
from tqdm import tqdm
from typing import Callable

from optim.trust_region import TrustRegionOptimizer
from optim.criterion import DualCriterion
from optim.blk import data_to_blocks, Block, NormalBlocks, OptimizationBlocks
from kernels.kernel import KernelBase
from optim.criterion import LogLoss

class Joker:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype,
                 kernel: KernelBase, criterion: DualCriterion,
                 data_blksz=2048, opt_blksz=512, device="cuda:0", incore=True, **kwargs):
        self.device = device
        self.dtype = dtype

        # try to move the data to the device
        if incore:
            try:
                self.x = x.to(self.device)
                self.y = y.to(self.device)
            except RuntimeError as e:
                print("msg: ", e)
                print("In-core mode failed, switching to out-of-core mode")
                self.x = x
                self.y = y
        else:
            self.x = x
            self.y = y

        self.n_train = x.size(0)
        self.n_target = 1 if len(y.size()) == 1 else y.size(1)

        # blocking the data
        self.opt_blocks  = data_to_blocks(self.x, self.y, blksz=opt_blksz,  blk_type="optim")
        self.data_blocks = data_to_blocks(self.x, self.y, blksz=data_blksz, blk_type="normal")
        self.data_blksz  = data_blksz
        self.opt_blksz   = opt_blksz

        # set the criterion and kernel
        self.criterion = criterion
        self.kernel = kernel
        self.c = criterion.c

        if isinstance(criterion, LogLoss):
            self.alpha = torch.rand((self.n_train, self.n_target), dtype=dtype, device=device) * criterion.c
            self.alpha = self.alpha * self.y.to(device=device, dtype=dtype)

            if type(self) == Joker:
                print("Reminder: Exact model used. Initialize gradient lookup beta...")
                self.beta = torch.zeros_like(self.alpha, dtype=dtype, device=device)
                for blk in self.data_blocks:
                    blk_ind = blk.blk_ind.to(device=device)
                    blk_data = blk.data.to(device=device, dtype=dtype)
                    self._update_beta(torch.zeros_like(self.alpha[blk_ind], device=device, dtype=dtype), self.alpha[blk_ind], blk_data)
        else:
            self.alpha = torch.zeros((self.n_train, self.n_target), dtype=dtype, device=device)
            self.beta  = torch.zeros_like(self.alpha, dtype=dtype, device=device)
        

        # set the prediction function and gram function
        self.pred_fcn  = self._eval_fcn
        self.gram_fcn  = lambda x: self.kernel(x, x)
        
        # set the optimizer
        self.optim = TrustRegionOptimizer()

        print("Model Initialization ready:")
        print(f"Kernel: {kernel}")
        print(f"The block size for optimization: {self.opt_blksz}")
        print(f"Start fitting with criterion: {self.criterion}, dtype: {self.dtype}")
        print(f"Device: {self.device}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, y: torch.Tensor = None):
        if y is None:
            y = torch.zeros((x.size(0), self.n_target), dtype=self.dtype, device=self.device) # make a dummy y
        test_blocks = data_to_blocks(x, y, blksz=self.data_blksz, blk_type="normal")
        return self.predict_blk(test_blocks)
    
    @torch.no_grad()
    def gram_and_grad(self, blk: Block):
        blk_data = blk.data.to(self.device, dtype=self.dtype)
        K_blk  = self.gram_fcn(blk_data)
        K_grad = self.kgrad(blk)
        return K_blk, K_grad
    
    @torch.no_grad()
    def kgrad(self, blk: Block):
        blk_ind = blk.blk_ind.to(self.device)
        return self.beta[blk_ind]
    
    @torch.no_grad()
    def predict_blk(self, data_blk: NormalBlocks):
        y_pred = torch.zeros((data_blk.data_num, self.n_target), device=self.device, dtype=self.dtype)
        for blk in data_blk:
            blk_data = blk.data.to(self.device, dtype=self.dtype)
            blk_ind  = blk.blk_ind.to(self.device)
            y_pred[blk_ind] = self.pred_fcn(blk_data)
        return y_pred

    @torch.no_grad()
    def fit(self, max_iter: int, max_iter_subprob: int, 
            max_region_size: float = 1024.0, region_shrink_freq: int = 1000, region_shrink_rate: float = 0.5,
            blk_strategy: str = "random",
            val_x: torch.Tensor = None, val_y: torch.Tensor = None, metric: Callable = None,
            verbose_freq: int = 500, verbose_primal_dual: bool = False, # default turn off verbose primal-dual because it's time consuming
            **kwargs):
        print("Start fitting...")
        print(f"block selection strategy: {blk_strategy}.")
        
        if val_x is not None and val_y is not None:
            val_blk = data_to_blocks(val_x, val_y, blksz=self.data_blksz, blk_type="normal")
        else:
            val_blk = None
        
        if region_shrink_rate > 1:
            raise ValueError("region_shrink_rate should not be larger than 1")
        
        pbar = tqdm(range(max_iter))
        for e in pbar:
            self._fit_one_block(blk_strategy, max_iter_subprob, max_region_size, **kwargs)
            
            # output the training process
            if (e+1) % verbose_freq == 0:
                verbose_item = {}
                # verbose the validation set 
                if val_blk is not None:
                    loss, val_metric = self.validation(val_blk, metric)
                    if metric is not None:
                        verbose_item.update(loss=f"{loss:.4f}", **val_metric)
                    else:
                        verbose_item.update(loss=f"{loss:.4f}")

                # verbose the primal and dual objective
                if verbose_primal_dual:
                    primal, dual = self.eval_primal_dual()
                    verbose_item.update(obj_p=f"{primal:.4f}", obj_d=f"{dual:.4f}")
                pbar.set_postfix(**verbose_item)
                print(".")

            # shrink the turst region size
            if (e+1) % region_shrink_freq == 0:
                max_region_size *= region_shrink_rate
        pbar.close()

    @torch.no_grad()
    def validation(self, val_blk: NormalBlocks, metric: Callable = None):
        loss = 0.0
        for blk in val_blk:
            blk_data = blk.data.to(self.device, dtype=self.dtype)
            blk_label = blk.label.to(self.device, dtype=self.dtype)
            y_pred = self.pred_fcn(blk_data)
            local_loss = self.criterion.primal_obj(y_pred, blk_label.to(y_pred.device)).item()
            loss += local_loss
        loss /= (val_blk.data_num * self.c)

        if metric is not None:
            val_y_pred = self.predict_blk(val_blk)
            val_metric = metric(val_y_pred)
        else:
            val_metric = {}

        return loss, val_metric
    
    @torch.no_grad()
    def eval_primal_dual(self):
        primal = 0
        dual   = 0
        for blk in self.data_blocks:
            blk_label = blk.label.to(self.device, dtype=self.dtype)
            y_pred = self.kgrad(blk)
            local_loss = self.criterion.primal_obj(y_pred, blk_label).sum().item()
            local_dual = self.criterion(self.alpha[blk.blk_ind], blk_label).sum().item()
            primal += local_loss
            dual   += local_dual

        regu_term = self._regu_term()
        primal += regu_term
        dual   += regu_term

        return primal, -dual

    @torch.no_grad()
    def _eval_fcn(self, x: torch.Tensor):
        res = torch.zeros((x.size(0), self.n_target), device=self.device, dtype=self.dtype)
        x = x.to(self.device, dtype=self.dtype)
        for blk in self.data_blocks:
            blk_ind = blk.blk_ind.to(self.device)
            blk_data = blk.data.to(self.device, dtype=self.dtype)
            res += self.kernel(x, blk_data) @ self.alpha[blk_ind]
        return res

    @torch.no_grad()
    def _update_beta(self, alpha: torch.Tensor, new_alpha: torch.Tensor, blk_data: torch.Tensor):
        for x_blk in self.data_blocks:
            blk_ind = x_blk.blk_ind.to(self.device)
            x = x_blk.data.to(self.device, dtype=self.dtype)
            self.beta[blk_ind] += self.kernel(x, blk_data) @ (new_alpha - alpha)

    @torch.no_grad()
    def _update_model(self, new_alpha: torch.Tensor, blk: Block):
        blk_ind = blk.blk_ind.to(self.device)
        blk_data = blk.data.to(self.device, dtype=self.dtype)
        self._update_beta(self.alpha[blk_ind], new_alpha, blk_data)
        self.alpha[blk_ind] = new_alpha

    @torch.no_grad()
    def _fit_one_block(self, blk_strategy: str, max_iter_subprob: int, max_region_size, **kwargs):
        if blk_strategy == "random":
            blk = self.opt_blocks.random_pick()
        elif blk_strategy == "cylic":
            blk = self.opt_blocks.next()
        else:
            raise NotImplementedError("Unsupport block strategy")
        
        # fetch block data
        blk_ind = blk.blk_ind.to(self.device)
        blk_label = blk.label.to(self.device, dtype=self.dtype)

        # prepare block kernel matrix and gradient
        K_blk, K_grad = self.gram_and_grad(blk)
        K_grad_rest = K_grad - K_blk @ self.alpha[blk_ind]

        # solve the block subproblem
        new_alpha = self.optim.minimize(self.criterion, K_blk, K_grad_rest, self.alpha[blk_ind], blk_label, 
                                        max_iter=max_iter_subprob,
                                        max_region_size=max_region_size,
                                        **kwargs)
        # update model parameters
        self._update_model(new_alpha, blk)

    @torch.no_grad()
    def _regu_term(self):
        return 0.5 * (self.alpha * self.beta).sum().item()
    
class InexactJoker(Joker):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype,
                 kernel: KernelBase, criterion: DualCriterion, n_features: int,
                 data_blksz=8192, opt_blksz=1024, device="cuda:0", feature_type="rff", **kwargs):
        super().__init__(x, y, dtype, kernel, criterion, data_blksz, opt_blksz, device)
        
        self.kernel.init_inexact(x.size(1), n_features, inexact_type=feature_type, device=device, dtype=dtype)

        if isinstance(criterion, LogLoss):
            # make beta = \phi(X) @ alpha
            self.beta = torch.zeros((n_features, self.n_target), dtype=dtype, device=device)
            for blk in self.data_blocks:
                blk_ind = blk.blk_ind.to(device=device)
                blk_data = blk.data.to(device=device, dtype=dtype)
                self._update_beta(torch.zeros_like(self.alpha[blk_ind], device=device, dtype=dtype), self.alpha[blk_ind], blk_data)
        else:
            self.beta = torch.zeros((n_features, self.n_target), dtype=dtype, device=device)
        
        # redefine the kernel evaluation functions
        self.pred_fcn = self._eval_fcn
        self.gram_fcn = self._gram_fcn
        print(f"Reminder: nexact model used with feature_type: {feature_type}, n_features: {n_features}.")

    @torch.no_grad()
    def kgrad(self, blk: Block):
        blk_data = blk.data.to(self.device, dtype=self.dtype)
        return self.pred_fcn(blk_data)

    @torch.no_grad()
    def gram_and_grad(self, blk: Block):
        blk_data = blk.data.to(self.device, dtype=self.dtype)
        feat_blk = self.feature(blk_data)
        K_blk  = feat_blk @ feat_blk.t()
        K_grad = feat_blk @ self.beta
        return K_blk, K_grad

    @torch.no_grad()
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_inexact_map(x)

    @torch.no_grad()
    def _eval_fcn(self, x: torch.Tensor):
        return self.feature(x) @ self.beta

    @torch.no_grad()
    def _gram_fcn(self, x: torch.Tensor) -> torch.Tensor:
        kzx_blk = self.feature(x)
        return kzx_blk.mm(kzx_blk.t())
    
    @torch.no_grad()
    def _update_beta(self, alpha: torch.Tensor, new_alpha: torch.Tensor, blk_data: torch.Tensor):
        delta = self.feature(blk_data).t() @ (new_alpha - alpha)
        self.beta += delta
    
    @torch.no_grad()
    def _regu_term(self):
        return 0.5 * (self.beta * self.beta).sum().item()