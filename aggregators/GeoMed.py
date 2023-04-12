import tools
import math
import torch
from geom_median.torch import compute_geometric_median 

# ---------------------------------------------------------------------------- #
class geomed(object):
    def __init__(self):
        self.name = "GeoMed"

    def aggregate(self, gradients, f=10, **kwargs):
        device = gradients[0][0].device
        n = len(gradients)
        grads = torch.stack(gradients)
        weights = torch.ones(n).to(device)
        gw = compute_geometric_median(gradients, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(grads-gw, dim=1)))
            gw = compute_geometric_median(gradients, weights).median

        return gw, [1], 0.0