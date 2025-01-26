from joker import Joker, InexactJoker
import torch
import kernels.kernel as ke
from optim.criterion import *
import numpy as np
# from task.task import make_task

from run_config import RuntimeConfig

run_config = RuntimeConfig()

cfg = run_config.full_config()
diejob = run_config.get_task()

X_train, y_train, X_test, y_test = diejob.load_data()
n_sub = 10000
sub_Xtrain = X_train[:n_sub]
sig2 = torch.mean(torch.pdist(sub_Xtrain) ** 2) # median trick

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

torch.manual_seed(42)
criterion = run_config.get_criterion()
kernel = run_config.get_kernel(gamma=1/sig2)

# random fourier feature is prioritized over fastfood
if cfg["n_rff"] > 0:
    # Random Fourier Feature, the most concise option for GPU computation. 
    n_rff = cfg["n_rff"]
    model = InexactJoker(X_train, y_train, dtype=cfg["dtype"], kernel=kernel, criterion=criterion, n_features=n_rff,
                         device=cfg["device"],
                         opt_blksz=cfg["blksz"],
                         data_blksz=cfg["data_blksz"],
                         incore=cfg["incore"],
                         inexact_type="rff")
elif cfg["n_fastfood"] > 0:
    # Fastfood, better option when the dimension of the data is high d~10^5 (O(nlogd) time and O(n) space)
    n_ff = cfg["n_fastfood"]
    model = InexactJoker(X_train, y_train, dtype=cfg["dtype"], kernel=kernel, criterion=criterion, n_features=n_ff,
                         device=cfg["device"],
                         opt_blksz=cfg["blksz"],
                         data_blksz=cfg["data_blksz"],
                         incore=cfg["incore"],
                         inexact_type="fastfood")
else:
    # Exact model
    model = Joker(X_train, y_train, dtype=cfg["dtype"], kernel=kernel, criterion=criterion,
                  device=cfg["device"],
                  opt_blksz=cfg["blksz"],
                  incore=cfg["incore"],
                  data_blksz=cfg["data_blksz"])

# Start training
model.fit(max_iter=cfg["max_iter"],
        max_iter_subprob=cfg["max_iter_subprob"],
        max_region_size=cfg["max_trust_region_size"],
        verbose_freq=cfg["verbose_freq"],
        region_shrink_freq=cfg["region_shrink_freq"],
        region_shrink_rate=cfg["region_shrink_rate"],
        do_precond=cfg["do_precond"],
        blk_strategy=cfg["blk_strategy"],
        val_x=X_test,
        val_y=y_test,
        metric=diejob.metric,
        verbose_primal_dual=True)


# Prediction
y_pred = model.predict(X_test, y_test)

# Evaluation
eval_res = diejob.metric(y_pred)
print(eval_res)

run_config.collect_result(eval_res)
run_config.summary()