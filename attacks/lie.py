# coding: utf-8

import math
import torch

# ---------------------------------------------------------------------------- #

def little_is_enough_attack(byz_grads, benign_grads, *args, know=True, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    ## attack strategy
    est_grads = torch.stack(byz_grads + benign_grads)
    if not know:
        est_grads = torch.stack(byz_grads)
    mu = torch.mean(est_grads, dim=0)
    sigma = torch.std(est_grads, dim=0)
    z = 1.0  # 0.3, 1.5 #Pre-calculated value for z_{max} from z-table, based on n=50, m=24 (and hence, s=2)
    lie_grad = mu - z * sigma

    return [lie_grad] * num_byzs
