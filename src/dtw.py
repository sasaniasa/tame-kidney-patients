import numpy as np
from tqdm import tqdm

inf = 1e10  # Large value for initialization

# Distance function between two sequences
def dist_func(s1, s2):
    s1 = s1.reshape((s1.shape[0], 1, s1.shape[1]))
    s2 = s2.reshape((1, s2.shape[0], s2.shape[1]))
    dist = np.abs(s1 - s2)
    dist = dist * dist
    dist = dist.mean(2)
    dist = np.sqrt(dist)
    return dist

# Dynamic Time Warping algorithm
def dtw(dist_mat, path, i=0, j=0):
    if path[i, j, 0] > -inf:
        return path[i, j]
    
    if i == dist_mat.shape[0] - 1 and j == dist_mat.shape[1] - 1:
        avg, dist_sum, steps = 0, 0, 0
    elif i == dist_mat.shape[0] - 1:
        avg, dist_sum, steps = dtw(dist_mat, path, i, j + 1)
    elif j == dist_mat.shape[1] - 1:
        avg, dist_sum, steps = dtw(dist_mat, path, i + 1, j)
    else:
        avg, dist_sum, steps = dtw(dist_mat, path, i + 1, j)
        for x in [dtw(dist_mat, path, i, j + 1), dtw(dist_mat, path, i + 1, j + 1)]:
            if avg > x[0]:
                avg, dist_sum, steps = x

    return_dist = dist_mat[i, j] + dist_sum
    return_steps = steps + 1
    return_avg = return_dist / return_steps

    path[i, j] = return_avg, return_dist, return_steps
    return path[i, j]

def dtw_iterative(dist_mat):
    n, m = dist_mat.shape
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_mat[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1],     # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    total_dist = dtw_matrix[n, m]
    steps = n + m  # or use actual path length if you need
    avg = total_dist / steps
    return avg, total_dist, steps

# Compute DTW distance between two patients
def compute_dtw(dist_mat, path, h, hi, hj):
    avg, dist_sum, steps = dtw_iterative(dist_mat)
    h[hi, hj] = avg
    h[hj, hi] = avg  # symmetric
    return avg