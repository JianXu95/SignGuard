import tools
import math
import torch

# ---------------------------------------------------------------------------- #
# NaN-resilient, coordinate-wise median GAR
class median(object):
    def __init__(self):
        self.name = "Median"

    def aggregate(self, gradients, f=10, **kwargs):
        n = len(gradients)
        return torch.stack(gradients).median(dim=0)[0], [i for i in range(n)], 0

