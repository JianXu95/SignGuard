import tools
import math
import numpy as np
import torch
# ---------------------------------------------------------------------------- #
# Multi-Krum GAR
class krum(object):
    def __init__(self):
        self.name = "Krum"

    def aggregate(self, gradients, f=10, m=None, **kwargs):
        n = len(gradients)
        grads = torch.stack(gradients)
        # Defaults
        if m is None:  # m = 1 --> krum, m = n --> mean
            m = 1
        # Compute all pairwise distances
        distances = tools.pairwise_distance2_faster(grads)
        # Compute the scores
        scores = list()
        for i in range(n):
            # Collect the distances
            grad_dists = list()
            for j in range(i):
                grad_dists.append(distances[i][j].item())
            for j in range(i + 1, n):
                grad_dists.append(distances[i][j].item())
            # Select the n - f - 1 smallest distances
            grad_dists.sort()
            scores.append((sum(grad_dists[:n - f - 1]), gradients[i], i))
        # Compute the average of the selected gradients
        scores.sort(key=lambda x: x[0])
        grad = sum(grad for _, grad, i in scores[:m]).div_(m)
        select_idx = [i for _, grad, i in scores[:m]]
        byz_num = (np.array(select_idx)<f).sum()

        return grad, select_idx, byz_num/f

class multi_krum(object):
    def __init__(self):
        self.name = "Multi-Krum"

    def aggregate(self, gradients, f=10, m=None, **kwargs):
        n = len(gradients)
        grads = torch.stack(gradients)
        # Defaults
        if m is None:  # m = 1 --> krum, m = n --> mean
            m = n - f - 2
        distances = tools.pairwise_distance2_faster(grads)
        # Compute the scores
        scores = list()
        for i in range(n):
            # Collect the distances
            grad_dists = list()
            for j in range(i):
                grad_dists.append(distances[i][j].item())
            for j in range(i + 1, n):
                grad_dists.append(distances[i][j].item())
            # Select the n - f - 1 smallest distances
            grad_dists.sort()
            scores.append((sum(grad_dists[:n - f - 1]), gradients[i], i))
        # Compute the average of the selected gradients
        scores.sort(key=lambda x: x[0])
        grad = sum(grad for _, grad, i in scores[:m]).div_(m)
        select_idx = [i for _, grad, i in scores[:m]]
        byz_num = (np.array(select_idx)<f).sum()

        return grad, select_idx, byz_num/f