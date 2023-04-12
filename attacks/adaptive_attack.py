# coding: utf-8

import math
import torch
import tools
from aggregators import aggregator
# ---------------------------------------------------------------------------- #

def adaptive_attack_std(byz_grads, benign_grads, agg_rule, g0=None, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    est_grads = torch.stack(byz_grads + benign_grads)
    benign_mean = torch.mean(est_grads, dim=0)
    # if not know:
    #     est_grads = torch.stack(byz_grads)
    grad_b = torch.mean(est_grads, dim=0)
    grad_p = -1.0 * torch.std(est_grads, dim=0)

    GAR = agg_rule
    if GAR.name in {'Multi-Krum', 'Bulyan', 'DnC', 'SignGuard'}:
        metric = num_byz_selected(num_byzs)
        max_byz_select = 1
    else:
        metric = dist2benign(num_byzs, benign_mean)
        max_byz_select = 0
    # search for optimal gamma
    # 0.75 for DnC maybe
    gamma_init = 2.0
    e = 0.01
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    last_metric = 0
    curr_metric = 0
    while abs(gamma_succ-gamma) > e and gamma > 0:
        grad_m = grad_b + gamma * grad_p
        grads = [grad_m]*num_byzs + benign_grads
        curr_metric = metric.output(grads, GAR, g0)
        if  curr_metric >= last_metric:
            last_metric = curr_metric
            gamma_succ = gamma
            gamma -= step/2
            if max_byz_select and curr_metric == 1.0:
                break
        else:
            gamma += step/2
        step = max(step / 2, 0.1)
    return [grad_b + gamma * grad_p] * num_byzs

def adaptive_attack_sign(byz_grads, benign_grads, agg_rule, g0=None, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    est_grads = torch.stack(byz_grads + benign_grads)
    benign_mean = torch.mean(est_grads, dim=0)
    # if not know:
    #     est_grads = torch.stack(byz_grads)
    grad_b = torch.mean(est_grads, dim=0)
    grad_p = -1.0 * torch.sign(grad_b)

    GAR = agg_rule
    if GAR.name in {'Multi-Krum', 'Bulyan', 'DnC', 'SignGuard'}:
        metric = num_byz_selected(num_byzs)
        max_byz_select = 1
    else:
        metric = dist2benign(num_byzs, benign_mean)
        max_byz_select = 0
    # search for optimal gamma
    # 0.75 for DnC maybe
    gamma_init = 0.5
    e = 0.01
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    last_metric = 0
    curr_metric = 0
    while abs(gamma_succ-gamma) > e and gamma > 0:
        grad_m = grad_b + gamma * grad_p
        grads = [grad_m]*num_byzs + benign_grads
        curr_metric = metric.output(grads, GAR, g0)
        if  curr_metric >= last_metric:
            last_metric = curr_metric
            gamma_succ = gamma
            gamma -= step/2
            if max_byz_select and curr_metric == 1.0:
                break
        else:
            gamma += step/2
        step = max(step / 2, 0.1)
    return [grad_b + gamma * grad_p] * num_byzs


def adaptive_attack_uv(byz_grads, benign_grads, agg_rule, g0=None, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    est_grads = torch.stack(byz_grads + benign_grads)
    benign_mean = torch.mean(est_grads, dim=0)
    # if not know:
    #     est_grads = torch.stack(byz_grads)
    grad_b = torch.mean(est_grads, dim=0)
    grad_p = -1.0*grad_b/torch.norm(grad_b)

    GAR = agg_rule
    if GAR.name in {'Multi-Krum', 'Bulyan', 'DnC', 'SignGuard'}:
        metric = num_byz_selected(num_byzs)
        max_byz_select = 1
    else:
        metric = dist2benign(num_byzs, benign_mean)
        max_byz_select = 0
    # search for optimal gamma
    # 0.75 for DnC maybe
    gamma_init = 5.0*torch.norm(grad_b) #0.3
    e = 0.01
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    last_metric = 0
    curr_metric = 0
    while abs(gamma_succ-gamma) > e and gamma > 0:
        grad_m = grad_b + gamma * grad_p
        grads = [grad_m]*num_byzs + benign_grads
        curr_metric = metric.output(grads, GAR, g0)
        if  curr_metric >= last_metric:
            last_metric = curr_metric
            gamma_succ = gamma
            gamma -= step/2
            if max_byz_select and curr_metric == 1.0:
                break
        else:
            gamma += step/2
        step = max(step / 2, 0.1)
    return [grad_b + gamma * grad_p] * num_byzs


class num_byz_selected():
    def __init__(self, num_byzs):
        self.metric = "Maximize the Number of Selected Byzantines"
        self.num_byzs = num_byzs

    def output(self, gradient, agg_rule, g0):
        agg_grad, benign_idx, byz_num = agg_rule.aggregate(gradient, f=self.num_byzs, g0=g0)
        return byz_num


class dist2benign():
    def __init__(self, num_byzs, benign_mean):
        self.metric = "Maximize the distance between Aggregated Gradient and benign Mean"
        self.num_byzs = num_byzs
        self.benign_mean = benign_mean

    def output(self, gradient, agg_rule, g0):
        benign_mean = self.benign_mean
        agg_grad, benign_idx, byz_num = agg_rule.aggregate(gradient, f=self.num_byzs, g0=g0)
        dist = torch.norm((agg_grad - benign_mean))
        return dist







