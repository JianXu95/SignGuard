# coding: utf-8

import math
import torch

# ---------------------------------------------------------------------------- #
# naive gradient attacks

def nan_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    # Generate the non-finite Byzantine gradient
    nan_grad = torch.empty_like(byz_grads[0]).to(device)
    nan_grad.copy_(torch.tensor((math.nan,), dtype=nan_grad.dtype))
    return [nan_grad] * num_byzs


def zero_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    # Return this Byzantine gradient 'num_byzs' times
    return [torch.zeros_like(byz_grads[0]).to(device)] * num_byzs


def random_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    grads = torch.stack(byz_grads)
    rand_grads = []
    for i in range(num_byzs):
        rand_grads.append(0.5*torch.randn(byz_grads[0].size()).to(device))
    return rand_grads


def noise_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    grads = torch.stack(byz_grads)
    device = byz_grads[0][0].device
    noisy_grads = []
    for i in range(num_byzs):
        noisy_grads.append(byz_grads[i] + 0.5 * torch.randn(byz_grads[0].size()).to(device))
    return noisy_grads


def signflip_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    return [-1.0*x for x in byz_grads]


def non_attack(byz_grads, *args, **kwargs):
    num_byzs = len(byz_grads)
    if num_byzs == 0:
        return list()
    return byz_grads