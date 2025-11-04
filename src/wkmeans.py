# coding=utf8
import numpy as np

inf = 100000000.0

def compute_weight(dist_mat, groups):
    weights = []
    for g in groups:
        dist_g = dist_mat[g][:, g]
        dist_avg = dist_g.mean(0)
        w = 1 / (1 + np.exp(dist_avg)) 
        w = w / w.sum()
        weights.append(w)
    return weights


def wkmeans_epoch(dist_mat, groups):

    assert dist_mat.min() >= 0
    weights = compute_weight(dist_mat, groups)

    cluster_dist = []
    for ig,g in enumerate(groups):
        dist = dist_mat[g]
        w = weights[ig]
        dist_avg = np.dot(w, dist)
        cluster_dist.append(dist_avg)

    new_groups = [[] for _ in groups]
    for i in range(len(dist_mat)):
        dist_i = [d[i] for d in cluster_dist]
        mind = min(dist_i)
        new_groups[dist_i.index(mind)].append(i)

    groups = new_groups
    return groups