# coding: utf-8

import math
import torch
import tools
# ---------------------------------------------------------------------------- #

def minmax_attack(byz_grads, benign_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    est_grads = torch.stack(byz_grads + benign_grads)
    grad_b = torch.mean(est_grads, dim=0)
    grad_p = -1.0 * torch.std(est_grads, dim=0)

    # search for optimal gamma
    gamma_init = 2
    e = 0.02
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    while abs(gamma_succ-gamma) > e:
        grad_m = grad_b + gamma * grad_p
        max_dist_m = tools.max_distance(est_grads, grad_m)
        max_dist_b = tools.max_pairwise_distance(est_grads)
        if  max_dist_m <= max_dist_b :
            gamma_succ = gamma
            gamma += step/2
        else:
            gamma -= step/2
        step = max(step/2, 0.1)

    return [grad_b + gamma_succ * grad_p] * num_byzs


def minsum_attack(byz_grads, benign_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    est_grads = torch.stack(byz_grads + benign_grads)
    grad_b = torch.mean(est_grads, dim=0)
    grad_p = -1.0 * torch.std(est_grads, dim=0)

    # search for optimal gamma
    gamma_init = 2
    e = 0.02
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    grad_m = grad_b + gamma * grad_p
    while abs(gamma_succ-gamma) > e:
        grad_m = grad_b + gamma * grad_p
        max_dist_m = tools.sum_distance(est_grads, grad_m)
        max_dist_b = tools.max_sum_distance(est_grads)
        if  max_dist_m <= max_dist_b :
            gamma_succ = gamma
            gamma += step/2
        else:
            gamma -= step/2
        step = max(step/2, 0.1)

    # return the malicious grads with optimal gamma
    return [grad_b + gamma_succ * grad_p] * num_byzs


