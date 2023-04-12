# coding: utf-8

import tools
import math
import torch
import numpy as np

# ---------------------------------------------------------------------------- #
# Divide-and-Conquer
class divide_conquer(object):
    def __init__(self):
        self.name = "DnC"

    def aggregate(self, gradients, f=10, **kwargs):
        """
        """
        device = gradients[0][0].device
        num_users = len(gradients)
        num_byzs = f
        c = (num_users-1)/num_users
        all_set = set([i for i in range(num_users)])
        grads = torch.stack(gradients, dim=0)
        num_param = grads.shape[1]
        all_idxs = [i for i in range(num_param)]
        # grads[torch.isnan(grads)] = 0  # remove nan

        iters = 1
        num_spars = 1000
        benign_idx = all_set
        for it in range(iters):
            idx = np.random.choice(all_idxs, num_spars, replace=False)
            # set of gradients subsampled using indices in idx
            gradss = grads[:, idx]
            # Compute mean of input gradients
            mu = torch.mean(gradss, dim=0)
            # get centered input gradients
            gradss_c = gradss - mu
            # get the top right singular eigenvector
            U, S, V = torch.svd(gradss_c)
            v = V[:, 0]
            # Compute outlier scores
            s = torch.mul((gradss - mu), v).sum(dim=1)**2
            dnc_idx = s.topk(int(num_users-c*num_byzs), dim=0, largest=False)[-1].cpu().numpy()
            benign_idx = benign_idx.intersection(set(dnc_idx))

        benign = list(benign_idx)
        byz_idx = (np.array(benign_idx) < f).sum()

        return grads[benign_idx].mean(dim=0), benign_idx, byz_idx/f

