# coding: utf-8

from numpy.core.fromnumeric import partition
import tools
import math
import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
import time
import matplotlib.pyplot as plt
from itertools import cycle
import copy
import random

# ---------------------------------------------------------------------------- #
# sign based detector, using clustering to remove outlier

class signguard_multiclass(object):
    def __init__(self):
        self.name = "SignGuard"

    def aggregate(self, gradients, f=10, epoch=1, g0=None, iteration=1, **kwargs):
        """
        """
        device = gradients[0][0].device
        num_users = len(gradients)
        all_set = set([i for i in range(num_users)])
        iters = 1
        grads = torch.stack(gradients, dim=0)
        grads[torch.isnan(grads)] = 0 # remove nan

        # gradient norm-based clustering
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)
        benign_idx1 = all_set
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))

        ## sign-gradient based clustering
        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)
        benign_idx2 = all_set

        dbscan = 0
        meanshif = int(1-dbscan)

        for it in range(iters):
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            gradss = grads[:, idx:(idx+num_spars)]
            sign_grads = torch.sign(gradss)
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)

            feat = [pos_feat, zero_feat, neg_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            # 
            if dbscan:
                clf_sign = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
            else:
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=50)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))

        benign_idx = list(benign_idx2.intersection(benign_idx1))
        byz_num = (np.array(benign_idx)<f).sum()

        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads/grad_norm)*grad_norm_clipped
        
        global_grad = grads_clip[benign_idx].mean(dim=0)

        return global_grad, benign_idx, byz_num/f


class signguard_multiclass_plus1(object):
    def __init__(self):
        self.name = "SignGuard-Sim"
        self.last_gradient = None

    def aggregate(self, gradients, f=10, epoch=1 , g0=None, iteration=1, **kwargs):
        """
        """
        device = gradients[0][0].device
        num_users = len(gradients)
        all_set = set([i for i in range(num_users)])
        iters = 1
        grads = torch.stack(gradients, dim=0)
        grads[torch.isnan(grads)] = 0 # remove nan

        # gradient norm-based clustering
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)
        benign_idx1 = all_set
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))

        ## sign-gradient based clustering
        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)
        benign_idx2 = all_set

        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads/grad_norm)*grad_norm_clipped
        refer_grad = grads_clip.mean(dim=0)

        if epoch < 3:
            grads_sim = tools.pairwise_similarity_faster(grads).median(dim=1)[0]
        else:
            last_grads = self.last_gradient
            grads_sim = torch.cosine_similarity(grads, last_grads, dim=-1)
        sim_feat = grads_sim.squeeze(dim=-1)
        sim_med = sim_feat.median(dim=0)[0]
        sim_feat = torch.clamp(sim_feat, 0, sim_med.item(), out=None) / (5 * sim_med + 1e-8)

        dbscan = 0
        meanshif = int(1-dbscan)

        for it in range(iters):
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            gradss = grads[:, idx:(idx+num_spars)]
            sign_grads = torch.sign(gradss)
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)

            feat = [pos_feat, zero_feat, neg_feat, sim_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            if dbscan:
                clf_sign = DBSCAN(eps=0.1, min_samples=3).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
            else:
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=50)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))

        benign_idx = list(benign_idx2.intersection(benign_idx1))
        byz_num = (np.array(benign_idx)<f).sum()
        
        global_grad = grads_clip[benign_idx].mean(dim=0)
        self.last_gradient = global_grad

        return global_grad, benign_idx, byz_num/f


class signguard_multiclass_plus2(object):
    def __init__(self):
        self.name = "SignCheck-Dist"
        self.last_gradient = None

    def aggregate(self, gradients, f=10, epoch=1, g0=None, iteration=1, **kwargs):
        """
        """
        device = gradients[0][0].device
        num_users = len(gradients)
        all_set = set([i for i in range(num_users)])
        iters = 1
        grads = torch.stack(gradients, dim=0)
        grads[torch.isnan(grads)] = 0 # remove nan

        # gradient norm-based clustering
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)
        benign_idx1 = all_set
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))

        ## sign-gradient based clustering
        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)
        benign_idx2 = all_set

        if epoch < 1:
            grads_dists = tools.pairwise_distance_faster(grads)
            dist_sorted, _ = torch.sort(grads_dists)
            dist_score = dist_sorted[:,:num_users//2].sum(dim=1)
            grads_sim = dist_score.squeeze(dim=-1)
            grads_sim = grads_sim.squeeze(dim=-1)
        else:
            last_grads = self.last_gradient
            grads_sim = torch.norm(grads-last_grads,dim=1)
            
        sim_med = grads_sim.median(dim=0)[0]
        sim_feat = grads_sim.ge(1.1*sim_med).float()/5.0

        dbscan = 0
        meanshif = int(1-dbscan)

        for it in range(iters):
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            gradss = grads[:, idx:(idx+num_spars)]
            sign_grads = torch.sign(gradss)
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)

            feat = [pos_feat, zero_feat, neg_feat, sim_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            if dbscan:
                clf_sign = DBSCAN(eps=0.05, min_samples=3).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
            else:
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=num_users)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))

        benign_idx = list(benign_idx2.intersection(benign_idx1))
        byz_num = (np.array(benign_idx)<f).sum()

        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads/grad_norm)*grad_norm_clipped
        
        global_grad = grads_clip[benign_idx].mean(dim=0)
        self.last_gradient = global_grad

        return global_grad, benign_idx, byz_num/f