import torch

class mean(object):
    def __init__(self):
        self.name = "Mean"
        
    def aggregate(self, gradients, **kwargs):
        n =len(gradients)
        return torch.stack(gradients, dim=0).mean(dim=0), [i for i in range(n)], 1.0

# ---------------------------------------------------------------------------- #

class trimmed_mean(object):
    def __init__(self):
        self.name = "TrMean"

    def aggregate(self, gradients, f=10, **kwargs):
        n = len(gradients)
        d = gradients[0].shape[0]
        trim_ratio = 1.0*f/n
        trim_num = int(len(gradients) * trim_ratio)
        grads = torch.stack(gradients, dim=0)
        grads_sorted = grads.sort(dim=0)[0]
        grads_trimmed = grads_sorted[trim_num:-trim_num, :]

        return grads_trimmed.mean(dim=0), [i for i in range(n)], 1.0