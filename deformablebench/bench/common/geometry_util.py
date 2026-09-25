"""Geometry helpers (slim P1 subset).

Only the pure-numpy helpers needed by ``metric_utils.get_covered_area_from_particles``
are mounted for P1. The heavier point-cloud helpers
(chamfer_distance / rgbd_2_camera_pointcloud / o3d / sklearn) pull torch+open3d+
sklearn and are needed only for point-cloud observations (P3); they will be
mounted then, with their dependencies.
"""

from __future__ import annotations

import numpy as np


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)."""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y


def vectorized_range(start, end):
    """Return an array of NxD, iterating from the start to the end."""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype("int")
    return idxes
