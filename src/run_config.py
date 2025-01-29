import numpy as np
import pandas as pd
import torch
import os

import argparse
from   typing        import Dict

from task.task       import make_task
from optim.criterion import make_criterion
from kernels.kernel  import make_kernel
from static_config   import *

class RuntimeConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Joker')
        self._add_arguments()
        self.config = vars(self.parser.parse_args())
        self.eval_res = None

        self.config["device"] = torch.device(self.config["device"])
        self.config["dtype"] = torch.float64 if self.config["dtype"] == "float64" or self.config["dtype"] == "double" else torch.float32

    def _add_arguments(self):
        self.parser.add_argument('--no-transform', action="store_false", dest="do_transform", help='Whether to transform the data')
        self.parser.add_argument('--no-target_transform', action="store_false", dest="do_target_transform", help='Whether to transform the target data')
        self.parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "huber", "svm", "log", "svr"], help="Which criterion to use")
        self.parser.add_argument('--c', type=float, default=1.0, help='Penalty parameter C of the error term')
        self.parser.add_argument('--delta_huber', type=float, default=-1, help='Regularization parameter lambda')
        self.parser.add_argument('--eps_insensitive', type=float, default=-1, help='Regularization parameter lambda')
        self.parser.add_argument("--data_blksz", type=int, default=2048, help="The block size for data loading")
        self.parser.add_argument("--blksz", type=int, default=512, help="The block size for optimization")
        self.parser.add_argument("--blk_strategy", type=str, default="random", choices=["random", "cylic"], help="How to choose the block during optimization")
        self.parser.add_argument('--max_iter', type=int, default=20000, help='Maximum number of iterations for fitting')
        self.parser.add_argument('--max_iter_subprob', type=int, default=50, help='Maximum number of iterations for fitting')
        self.parser.add_argument("--max_trust_region_size", type=float, default=64, help="The maximal trust region size")
        self.parser.add_argument("--region_shrink_freq", type=int, default=1000, help="Frequency of region shrink")
        self.parser.add_argument("--region_shrink_rate", type=float, default=0.9, help="Rate of region shrink")
        self.parser.add_argument('--verbose_freq', type=int, default=5000, help='Frequency of verbose output during fitting')
        self.parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the code")
        self.parser.add_argument("--dtype", type=str, default="float32", help="Data type to use")
        self.parser.add_argument("--dataset", type=str, default="susy", choices=["susy", "msd", "higgs", "houseelec", "cifar5m"], help="Dataset to run the code")
        self.parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "student", "lap"], help="Kernel to use")
        self.parser.add_argument("--do_precond", action="store_true", help="If activated, preconditioning is used. Not implemented yet.")
        self.parser.add_argument("--n_rff", type=int, default=50000, help="Number of samples for RFF approximation, negative means full kernel")
        self.parser.add_argument("--n_fastfood", type=int, default=-1, help="Number of samples for Fastfood approximation, negative means full kernel")
        self.parser.add_argument("--no-incore", action="store_false", dest="incore", help="Whether place the data into GPU")
        self.parser.add_argument("--kernel-sig", type=float, default=-1, help="The sigma parameter in gaussian kernel. -1 denotes default (mean of pair-wise squared distance)")

    def get_task(self):
        return make_task(self.config["dataset"], do_transform=self.config["do_transform"], do_target_transform=self.config["do_target_transform"])
    
    def get_criterion(self):
        return make_criterion(self.config["criterion"], c=self.config["c"], delta=self.config["delta_huber"], eps=self.config["eps_insensitive"], dtype=self.config["dtype"])
    
    def get_kernel(self, gamma=1.0, d=2):
        if self.config["kernel_sig"] > 0:
            if self.config["kernel"] == "rbf":
                gamma = 0.5/((self.config["kernel_sig"]) ** 2)
            elif self.config["kernel"] == "lap":
                gamma = 1 / self.config["kernel_sig"]
        return make_kernel(self.config["kernel"], gamma=gamma, degree=d)

    def full_config(self):
        return self.config
    
    def collect_result(self, eval_res: Dict[str, str]):
        self.eval_res = eval_res

    def summary(self):
        if self.eval_res is None:
            raise RuntimeError("No evaluation results available")
        
        # join the config with the eval_res into a big dict
        summary = {**self.config, **self.eval_res, "machine": MACHINE_NAME}
        run_summary = pd.DataFrame(summary, index=[0])

        filename = f"log/new_{self.config['dataset']}_run_summary_{MACHINE_NAME}.csv"

        # create the log directory if it does not exist
        os.makedirs("log", exist_ok=True)
        
        # if there is no file, write the header
        if not os.path.exists(filename):
            run_summary.to_csv(filename, index=False)
        else:
            run_summary.to_csv(filename, index=False, mode='a', header=False)

        print(run_summary)

        return run_summary