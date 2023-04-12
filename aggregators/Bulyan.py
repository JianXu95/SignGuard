# coding: utf-8
###
 # Bulyan over Multi-Krum GAR.
###

import tools
import math
import torch
import numpy as np

# ---------------------------------------------------------------------------- #
# Bulyan GAR class
class bulyan(object):
    def __init__(self):
        self.name = "Bulyan"

    def aggregate(self, gradients, f=10, m=None, **kwargs):
        """ Bulyan over Multi-Krum rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          m         Optional number of averaged gradients for Multi-Krum
          ...       Ignored keyword-arguments
        Returns:
          Aggregated gradient
        """
        n = len(gradients)
        d = gradients[0].shape[0]  # dimension of param
        # Defaults
        m_max = n - f - 2
        if m is None:
            m = m_max
        # Compute all pairwise distances
        distances = list([(math.inf, None)] * n for _ in range(n))
        for gid_x, gid_y in tools.pairwise(tuple(range(n))):
            dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
            if not math.isfinite(dist):
                dist = math.inf
            distances[gid_x][gid_y] = (dist, gid_y)
            distances[gid_y][gid_x] = (dist, gid_x)
        # Compute the scores
        scores = [None] * n
        for gid in range(n):
            dists = distances[gid]
            dists.sort(key=lambda x: x[0])
            dists = dists[:m]
            scores[gid] = (sum(dist for dist, _ in dists), gid)
            distances[gid] = dict(dists)
        # Selection loop
        selected = torch.empty(n - 2 * f - 2, d, dtype=gradients[0].dtype, device=gradients[0].device)
        select_idx = []
        for i in range(selected.shape[0]):
            # Update 'm'
            m = min(m, m_max - i)
            # Compute the average of the selected gradients
            scores.sort(key=lambda x: x[0])
            selected[i] = sum(gradients[gid] for _, gid in scores[:m]).div_(m)
            select_idx.append(scores[0][1])
            # Remove the gradient from the distances and scores
            gid_prune = scores[0][1]
            scores[0] = (math.inf, None)
            for score, gid in scores[1:]:
                if gid == gid_prune:
                    scores[gid] = (score - distances[gid][gid_prune], gid)
        # Coordinate-wise averaged median
        m = selected.shape[0] - 2 * f
        median = selected.median(dim=0)[0]
        closests = selected.clone().sub_(median).abs_().topk(m, dim=0, largest=False, sorted=False).indices
        closests.mul_(d).add_(torch.arange(0, d, dtype=closests.dtype, device=closests.device))
        avgmed = selected.take(closests).mean(dim=0)
        # Return resulting gradient
        byz_num = (np.array(select_idx) < f).sum()
        return avgmed, select_idx, byz_num/f

# ---------------------------------------------------------------------------- #