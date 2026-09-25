import numpy as np
import cv2
from deformablebench.bench.common.geometry_util import vectorized_meshgrid, vectorized_range

def get_covered_area_from_particles(cloth_particles: np.ndarray, cloth_particle_radius: float) -> float:
    """
    Calculate the covered area by taking max x,y cood and min x,y
    coord, create a discritized grid between the points

    :param cloth_particles: Current positions of the particles in virtual coordinate system
    :param cloth_particle_radius: Radius of the particles
    """
    pos = cloth_particles
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 1]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]

def get_iou_from_2d_points(A: np.ndarray, B: np.ndarray, vis: bool = False):
    min_range = np.min(np.concatenate([np.min(A, axis=0), np.min(B, axis=0)], axis=0), axis=0)
    max_range = np.max(np.concatenate([np.max(A, axis=0), np.max(B, axis=0)], axis=0), axis=0)
    # Normalize the point sets to have x and y coordinates between 0 and 1
    A = (A - min_range) / (max_range - min_range)
    B = (B - min_range) / (max_range - min_range)

    # Scale the normalized point sets to have coordinates between 0 and 99
    A *= 99
    B *= 99

    # Create a blank image with dimensions (100, 100) for each set of points
    mask_A = np.zeros((100, 100), dtype=np.uint8)
    mask_B = np.zeros((100, 100), dtype=np.uint8)

    # Draw the points on the masks as white circles with radius 2
    for point in A:
        cv2.circle(mask_A, tuple(np.int32(point)), 2, 255, -1)
    for point in B:
        cv2.circle(mask_B, tuple(np.int32(point)), 2, 255, -1)

    if vis:
        cv2.imshow('mask A', mask_A)
        cv2.imshow('mask B', mask_B)
        cv2.waitKey()

    # Compute the intersection and union of the masks
    intersection = cv2.bitwise_and(mask_A, mask_B)
    union = cv2.bitwise_or(mask_A, mask_B)

    # Compute the Intersection over Union (IoU) metric
    iou = cv2.countNonZero(intersection) / float(cv2.countNonZero(union))
    return iou

def _rigid_transform_3D(A: np.ndarray, B: np.ndarray):
    """Least-squares rigid (rotation+translation) transform mapping A->B.
    A, B are 3xN. Reflections are corrected (proper rotation only)."""
    cA = A.mean(axis=1, keepdims=True)
    cB = B.mean(axis=1, keepdims=True)
    H = (A - cA) @ (B - cB).T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:            # reflection -> flip smallest singular axis
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ cA + cB
    return R, t


def _superimpose(current: np.ndarray, goal: np.ndarray, indices=None) -> np.ndarray:
    if indices is not None and int(np.sum(indices)) > 10:
        R, t = _rigid_transform_3D(current[indices].T, goal[indices].T)
    else:
        R, t = _rigid_transform_3D(current.T, goal.T)
    return (R @ current.T + t).T


def deformable_distance(goal_verts: np.ndarray, current_verts: np.ndarray,
                        max_coverage: float, deformable_weight: float = 0.65,
                        up_axis: int = 2):
    """ICP-aligned distance from a deformable's current state to its canonical
    (flat/goal) state. Port of ClothFunnels/UniFolding ``deformable_distance``
    (numpy-only; the open3d clouds there were viz-only). Lower == flatter.

    Both vert arrays must share vertex ordering (same mesh). The vertical
    ``up_axis`` is collapsed so the fit is in the table plane; the fit is
    flip-robust (min over goal and its mirror) and scale-invariant (divided by
    sqrt(max_coverage)). Returns (weighted, l2_icp, icp_reverse, real_l2)."""
    goal = np.asarray(goal_verts, dtype=float).copy()
    current = np.asarray(current_verts, dtype=float).copy()
    goal[:, up_axis] = 0.0
    current[:, up_axis] = 0.0
    mirror = 1 if up_axis == 0 else 0
    flipped = goal.copy()
    flipped[:, mirror] *= -1.0

    real_l2 = min(np.linalg.norm(goal - current, axis=1).mean(),
                  np.linalg.norm(flipped - current, axis=1).mean())

    thr = np.sqrt(max_coverage) * 0.3
    icp = _superimpose(current, goal)
    for _ in range(5):
        idx = np.linalg.norm(icp - goal, axis=1) < thr
        icp = _superimpose(icp, goal, idx)

    R, t = _rigid_transform_3D(goal.T, current.T)
    rev = (R @ goal.T + t).T
    idx = np.linalg.norm(rev - current, axis=1) < thr
    rev = _superimpose(rev, current, idx)

    l2 = min(np.linalg.norm(icp - goal, axis=1).mean(),
             np.linalg.norm(icp - flipped, axis=1).mean())
    icpd = min(np.linalg.norm(goal - rev, axis=1).mean(),
               np.linalg.norm(flipped - rev, axis=1).mean())

    s = np.sqrt(max_coverage)
    l2 /= s; icpd /= s; real_l2 /= s
    weighted = deformable_weight * l2 + (1.0 - deformable_weight) * icpd
    return float(weighted), float(l2), float(icpd), float(real_l2)


def flatten_metrics(current: np.ndarray, canonical: np.ndarray,
                    particle_radius: float = 0.005) -> dict:
    """All flatness signals for a cloth vs its canonical flat mesh:
      coverage         = footprint area / canonical footprint (↑ flatter)
      weighted_distance= ICP-to-canonical shape distance (↓ flatter, catches folds)
      l2/icp_distance  = the two components of weighted_distance
      iou              = footprint-shape IoU vs canonical (diagnostic; no rot-align)
    ``canonical`` must be the same mesh (vertex correspondence with ``current``)."""
    max_cov = get_covered_area_from_particles(canonical, particle_radius)
    coverage = get_covered_area_from_particles(current, particle_radius) / max_cov
    weighted, l2, icpd, real_l2 = deformable_distance(canonical, current, max_cov)
    iou = get_iou_from_2d_points(current[:, :2].copy(), canonical[:, :2].copy())
    return {"coverage": float(coverage), "weighted_distance": weighted,
            "l2_distance": l2, "icp_distance": icpd, "real_l2_distance": real_l2,
            "iou": float(iou)}


def check_is_reachable(target_pos: np.ndarray, base_pos: np.ndarray, workspace_radius:float=0.6) -> bool:
    if target_pos is None or base_pos is None:
        logger.warning("[Reachability Check] Target or base position is None")
        return False
    vec = target_pos[:3] - base_pos[:3]
    if np.linalg.norm(vec) < workspace_radius:
        return True
    else:
        return False