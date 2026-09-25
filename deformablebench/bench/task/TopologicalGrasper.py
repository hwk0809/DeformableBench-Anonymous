from dataclasses import dataclass
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Any, Generator, Tuple
from omegaconf import DictConfig
from easydict import EasyDict as edict
from loguru import logger
from scipy.spatial.transform import Rotation as R

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deformablebench.bench.task.base_task import BaseTask
from deformablebench.bench.common.metric_utils import get_covered_area_from_particles, check_is_reachable
from deformablebench.bench.common.visualize_utils import visualize_occlusion_check,visualize_cloth_topology_o3d

import trimesh
import networkx as nx
from scipy.spatial import cKDTree

import open3d as o3d


class TopologicalGrasper:
    def __init__(self, nodes_config: Dict, adjacency_graph: List[Tuple[str, str]],
                 obj_path: str = None,
                 jitter_enabled: bool = False,
                 jitter_radius: float = 0.025,
                 jitter_mode: str = "inward"):
        """
        Args:
            nodes_config: Dict mapping node-name → {index, core_radius, weight}.
            adjacency_graph: List of (node_name_a, node_name_b) edges allowed for
                bimanual grasp pairs.
            obj_path: Path to canonical cloth .obj used for mesh-distance computations.
            jitter_enabled: If True, after `selector_optimal_grasp_pair` picks the
                topologically optimal (l_idx, r_idx), each index is replaced by a
                nearby vertex within `jitter_radius` meters. Default False (legacy).
            jitter_radius: Radius (m) within which to look for jitter neighbors.
            jitter_mode: "inward" (only neighbors closer to cloth center, biased
                grasp toward interior; safer) or "uniform" (any neighbor in radius).
        """
        self.nodes_config = nodes_config
        # self.adjacency_graph = adjacency_graph
        # 双向
        self.adjacency_graph = set()
        for u, v in adjacency_graph:
            self.adjacency_graph.add((u, v))
            self.adjacency_graph.add((v, u))

        # 标准网格的位置
        self.mesh_vertices_pos = self.load_canonical_mesh(obj_path)
        self.safe_neighbor_map = None
        self.n_vertices = len(self.mesh_vertices_pos)
        self.n_nodes = len(self.nodes_config)

        # 预分配的节点权重和归属索引
        self.vertex_weights = np.zeros(self.n_vertices, dtype=np.float32)
        self.vertex_node_ids = np.full(self.n_vertices, -1, dtype=int)
        self.node_weights = np.zeros(self.n_nodes, dtype=np.float32)

        self.node_names = list(self.nodes_config.keys())
        self.name_to_index = {name: i for i, name in enumerate(self.node_names)}
        self._precompute_vertex_attributes()

        # Inward jitter (post-selection randomization to broaden grasp coverage)
        self._jitter_enabled = bool(jitter_enabled)
        self._jitter_radius = float(jitter_radius)
        if jitter_mode not in ("inward", "uniform"):
            raise ValueError(
                f"jitter_mode must be 'inward' or 'uniform', got {jitter_mode!r}"
            )
        self._jitter_mode = jitter_mode
        if self._jitter_enabled:
            logger.info(
                f"[TopologicalGrasper] inward_jitter ENABLED "
                f"(radius={self._jitter_radius:.3f}m, mode={self._jitter_mode})"
            )

    def load_canonical_mesh(self, obj_path: str, safe_radius: float = 0.025):
        """
        加载初始平铺网格并预计算邻域
        """
        mesh = trimesh.load(obj_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0] if mesh.geometry else trimesh.Trimesh()
        
        self.mesh_vertices_pos = mesh.vertices.copy()
        
        # 预计算邻域
        kdtree = cKDTree(self.mesh_vertices_pos)
        self.safe_neighbor_map = kdtree.query_ball_point(self.mesh_vertices_pos, r=safe_radius)
        
        logger.info(f"Loaded mesh: {len(self.mesh_vertices_pos)} vertices")
        logger.info(f"Avg neighbors: {np.mean([len(n) for n in self.safe_neighbor_map]):.1f}")
        
        return self.mesh_vertices_pos
    
    def _precompute_vertex_attributes(self):
        """
        [核心预处理]
        基于标准网格的几何信息，计算每个顶点的语义归属和权重。
        填充 self.vertex_weights 和 self.vertex_node_ids, 同时记录到节点方便调试
        """

        center_positions = []
        core_radii = []
        node_weights = [] 

        for name in self.node_names:
            cfg = self.nodes_config[name]
            center_positions.append(self.mesh_vertices_pos[cfg['index']])
            core_radii.append(cfg.get('core_radius', 0.05))
            node_weights.append(cfg.get('weight', 1.0))

        center_positions = np.array(center_positions)
        core_radii = np.array(core_radii)
        node_weights = np.array(node_weights)
        self.node_weights = node_weights

        # 使用 KDTree 查询每个顶点最近的节点中心，赋予归属和权重
        kdtree = cKDTree(center_positions)
        dists, nearest_indices = kdtree.query(self.mesh_vertices_pos)
        self.vertex_node_ids = nearest_indices
        
        #  计算是不是最适合抓取的核心顶点，以到核心节点距离划分
        thresholds = core_radii[nearest_indices]
        is_core_mask = dists <= thresholds

        self.vertex_weights[:]= 1.0

        # 给核心顶点赋值
        core_indices, core_weights = self._compute_core_node_weights(
           dists=dists,thresholds=thresholds,nearest_indices=nearest_indices,
           node_weights=self.node_weights, max_weight=1000.0, min_weight=100.0
        )
        if len(core_indices) > 0:
            self.vertex_weights[core_indices] = core_weights

        

    def _compute_core_node_weights(self,
                                   dists: np.ndarray,
                                   thresholds: np.ndarray,
                                   nearest_indices: np.ndarray,
                                   node_weights: np.ndarray,
                                   max_weight: float = 1000.0,
                                   min_weight: float = 100.0):
        """
        计算核心圈的权重分布
        Args:
            dists: 每个顶点到最近节点中心的距离 (N,)
            thresholds: 每个顶点对应的核心半径阈值 (N,)
            nearest_indices: 每个顶点最近的节点索引 (N,)
            node_weights: 各节点的权重系数 (M,)
            max_weight: 核心中心最大权重
            min_weight: 核心边界最小权重
        
        Returns:
            core_indices: 核心圈顶点索引 (K,)
            core_weights: 对应的权重值 (K,)
        """
        is_core_mask = dists <= thresholds
        core_indices = np.where(is_core_mask)[0]
        if len(core_indices) == 0:
            logger.warning("No vertices in core regions!")
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        normalized_dists = dists[core_indices] / thresholds[core_indices]
        
        # 线性衰减：核心中心权重最高，边界权重最低
        decay = 1 - normalized_dists

        core_weights = min_weight + (max_weight - min_weight) * decay

        # 应用节点权重系数
        core_weights *= node_weights[nearest_indices[core_indices]]

        return core_indices, core_weights

    @staticmethod
    def detect_cloth_topology(obj_path: str) -> edict:
        """
        【工具函数】自动检测布料拓扑结构（角点、边）
        
        Returns:
            EasyDict with clear structure:
                topo.corners = [c0_idx, c1_idx, c2_idx, c3_idx]
                topo.edges = [[e0_indices], [e1_indices], ...]
                topo.edge_centers = [ec0_idx, ec1_idx, ...]
                topo.vertices = np.ndarray (N, 3)
                topo.boundary = np.ndarray (ordered indices)
        """
        mesh = trimesh.load(obj_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0] if mesh.geometry else trimesh.Trimesh()
        
        vertices = mesh.vertices
        
        # 1. 提取边界
        boundary_edges = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse) == 1]
        boundary_vertices_idx = np.unique(boundary_edges.flatten())
        
        # 2. 构建有序环
        adj = {v: [] for v in boundary_vertices_idx}
        for u, v in boundary_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        ordered_boundary = []
        if len(boundary_vertices_idx) > 0:
            curr = boundary_vertices_idx[0]
            visited = set()
            while len(ordered_boundary) < len(boundary_vertices_idx):
                ordered_boundary.append(curr)
                visited.add(curr)
                next_node = next((n for n in adj[curr] if n not in visited), None)
                if next_node is None:
                    break
                curr = next_node
        ordered_boundary = np.array(ordered_boundary)
        
        # 3. 识别角点
        center = vertices[ordered_boundary].mean(axis=0)
        dists = np.linalg.norm(vertices[ordered_boundary] - center, axis=1)
        c1_idx = ordered_boundary[np.argmax(dists)]
        
        dists_c1 = np.linalg.norm(vertices[ordered_boundary] - vertices[c1_idx], axis=1)
        c3_idx = ordered_boundary[np.argmax(dists_c1)]
        
        idx_c1 = np.where(ordered_boundary == c1_idx)[0][0]
        idx_c3 = np.where(ordered_boundary == c3_idx)[0][0]
        
        if idx_c1 < idx_c3:
            seg1 = list(range(idx_c1, idx_c3 + 1))
            seg2 = list(range(idx_c3, len(ordered_boundary))) + list(range(0, idx_c1 + 1))
        else:
            seg1 = list(range(idx_c3, idx_c1 + 1))
            seg2 = list(range(idx_c1, len(ordered_boundary))) + list(range(0, idx_c3 + 1))
        
        vec_diag = vertices[c3_idx] - vertices[c1_idx]
        vec_diag /= np.linalg.norm(vec_diag)
        
        def get_farthest_from_line(indices, p_start):
            pts = vertices[ordered_boundary[indices]]
            cross_dist = np.linalg.norm(np.cross(pts - p_start, vec_diag), axis=1)
            return ordered_boundary[indices][np.argmax(cross_dist)]
        
        c2_idx = get_farthest_from_line(seg1, vertices[c1_idx])
        c4_idx = get_farthest_from_line(seg2, vertices[c1_idx])
        
        # 4. 按边界顺序排列
        raw_corners = [c1_idx, c2_idx, c3_idx, c4_idx]
        loop_indices = [np.where(ordered_boundary == c)[0][0] for c in raw_corners]
        corners = [p[1] for p in sorted(zip(loop_indices, raw_corners))]
        
        # 5. 切分边
        edges = []
        edge_centers = []
        for i in range(4):
            c_start = corners[i]
            c_end = corners[(i+1) % 4]
            
            idx_s = np.where(ordered_boundary == c_start)[0][0]
            idx_e = np.where(ordered_boundary == c_end)[0][0]
            
            if idx_s < idx_e:
                segment = ordered_boundary[idx_s : idx_e+1]
            else:
                segment = np.concatenate([ordered_boundary[idx_s:], ordered_boundary[:idx_e+1]])
            
            edges.append(segment.tolist())
            edge_centers.append(segment[len(segment)//2])
        
        return edict({
            'corners': corners,              # List[int]: 4个角点索引
            'edges': edges,                  # List[List[int]]: 4条边
            'edge_centers': edge_centers,    # List[int]: 4个边中心
            'vertices': vertices,            # np.ndarray: (N, 3)
            'boundary': ordered_boundary     # np.ndarray: 有序边界环
        })
    
    def get_reachable_indices(
        self, 
        vis_indices: np.ndarray, 
        cloth_pos: np.ndarray,
        base_left: np.ndarray = None,
        base_right: np.ndarray = None,
        workspace_radius: float = 0.75
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速筛选可见且可达的点索引
        
        Args:
            vis_indices: 可见点的索引数组
            cloth_pos: 当前布料所有顶点位置 (N, 3)
            base_left: 左臂基座位置 (3,) 或 (7,) with quaternion
            base_right: 右臂基座位置 (3,) 或 (7,) with quaternion
            workspace_radius: 工作空间半径
        
        Returns:
            (left_reachable_indices, right_reachable_indices)
        """
        vis_indices = np.asarray(vis_indices)
        
        if len(vis_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        
        # 提取可见点的位置
        vis_positions = cloth_pos[vis_indices]  # (M, 3)
        
        # 向量化计算到两个基座的距离
        if base_left is not None:
            base_left_pos = base_left[:3]  # 取xyz，忽略姿态
            dist_to_left = np.linalg.norm(vis_positions - base_left_pos, axis=1)  # (M,)
            left_reachable_mask = dist_to_left < workspace_radius
            left_reachable_indices = vis_indices[left_reachable_mask]
        else:
            left_reachable_indices = np.array([], dtype=int)
        
        if base_right is not None:
            base_right_pos = base_right[:3]
            dist_to_right = np.linalg.norm(vis_positions - base_right_pos, axis=1)  # (M,)
            right_reachable_mask = dist_to_right < workspace_radius
            right_reachable_indices = vis_indices[right_reachable_mask]
        else:
            right_reachable_indices = np.array([], dtype=int)
        
        return left_reachable_indices, right_reachable_indices

    def _jitter_grasp_index(self, idx: int, cloth_pos: np.ndarray) -> int:
        """Replace grasp index with a nearby vertex within `jitter_radius`.

        For mode='inward': only neighbors that are closer to the cloth center
        than the original (in xy plane) are kept, then one is sampled uniformly.
        For mode='uniform': any vertex within radius is a candidate.

        If no valid neighbors are found, returns the original index unchanged.
        """
        if idx is None or self._jitter_radius <= 0.0:
            return idx
        origin = cloth_pos[idx]

        # All vertices within jitter_radius (3D L2 distance in current cloth pose)
        dists = np.linalg.norm(cloth_pos - origin, axis=1)
        nearby_mask = dists <= self._jitter_radius
        nearby_mask[idx] = False  # exclude self
        if not np.any(nearby_mask):
            return idx
        nearby_indices = np.where(nearby_mask)[0]

        if self._jitter_mode == "inward":
            cloth_center = cloth_pos.mean(axis=0)
            d_origin_center = np.linalg.norm(origin[:2] - cloth_center[:2])
            d_nearby_center = np.linalg.norm(
                cloth_pos[nearby_indices, :2] - cloth_center[:2], axis=1
            )
            inward_mask = d_nearby_center < d_origin_center
            candidates = nearby_indices[inward_mask]
        else:  # uniform
            candidates = nearby_indices

        if len(candidates) == 0:
            return idx
        chosen = int(np.random.choice(candidates))
        logger.info(
            f"[TopologicalGrasper] jitter {idx} → {chosen} "
            f"(mode={self._jitter_mode}, radius={self._jitter_radius:.3f}, "
            f"|delta|={np.linalg.norm(cloth_pos[chosen] - origin)*1000:.1f}mm, "
            f"n_candidates={len(candidates)})"
        )
        return chosen

    def selector_optimal_grasp_pair(self,
                                    cloth_pos: np.ndarray,
                                    vis_indices: np.ndarray,
                                    base_left: np.ndarray,
                                    base_right: np.ndarray,
                                    min_safe_dist: float = 0.15,
                                    workspace_radius: float = 0.75,
                                    visualize: bool = False
                                  ) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:

        left_vis_reachable_indices, right_vis_reachable_indices = self.get_reachable_indices(
            vis_indices=vis_indices,
            cloth_pos=cloth_pos,
            base_left=base_left,
            base_right=base_right,
            workspace_radius=workspace_radius)
        
        l_indices = np.array(left_vis_reachable_indices)
        r_indices = np.array(right_vis_reachable_indices)
        best_total_score = -np.inf
        best_pair = (None, None)
        best_debug_info = edict()
        canonical_dist = None
        
        vec_base_direction = base_left[:3] - base_right[:3]

        # 遍历允许的拓扑连接边 (e.g., L_Hem <-> R_Hem)
        for node_a_name, node_b_name in self.adjacency_graph:
            idx_a = self.name_to_index.get(node_a_name)
            idx_b = self.name_to_index.get(node_b_name)

            if idx_a is None or idx_b is None: continue

            # 1. 快速掩码筛选：只看属于当前 Node A 和 Node B 的点
            l_mask = (self.vertex_node_ids[l_indices] == idx_a)
            r_mask = (self.vertex_node_ids[r_indices] == idx_b)

            curr_l_candidates = l_indices[l_mask]
            curr_r_candidates = r_indices[r_mask]


            if len(curr_l_candidates) == 0 or len(curr_r_candidates) == 0:
                continue

            # 提取权重
            w_l = self.vertex_weights[curr_l_candidates] # (M,)
            w_r = self.vertex_weights[curr_r_candidates] # (K,)
            
            # 提取位置 (Mesh Canonical Position) 用于计算展开距离
            p_l = self.mesh_vertices_pos[curr_l_candidates] # (M, 3)
            p_r = self.mesh_vertices_pos[curr_r_candidates] # (K, 3)

            # 物理距离
            p_phys_l = cloth_pos[curr_l_candidates]
            p_phys_r = cloth_pos[curr_r_candidates]
            
            # ---------- 矩阵评分 (Vectorized Scoring)-----------
            # 距离矩阵 (Mesh Canonical Distance)
            dist_matrix = np.linalg.norm(p_l[:, None, :] - p_r[None, :], axis=2)

            # 安全距离
            dist_phys_mat = np.linalg.norm(p_phys_l[:, None, :] - p_phys_r[None, :], axis=2)
            
            # 1. 总分 = 权重和 + 距离
            # 权重项(2000/1000/2) 决定了优先级层级，距离项(0.x) 决定了同层级内的优劣
            score_matrix = (w_l[:, None] + w_r[None, :]) + dist_matrix

            # 1.5 Proximity tiebreaker: 分数相同时优先选离机械臂近的角点
            dist_to_left_base = np.linalg.norm(p_phys_l - base_left[:3], axis=1)   # (M,)
            dist_to_right_base = np.linalg.norm(p_phys_r - base_right[:3], axis=1)  # (K,)
            score_matrix -= 0.01 * (dist_to_left_base[:, None] + dist_to_right_base[None, :])

            # 2 .物理熔断
            unsafe_mask = dist_phys_mat < min_safe_dist
            score_matrix[unsafe_mask] = -np.inf
            
            # 3. 交叉臂熔断
            vec_grasp_mat = p_phys_l[:, None, :] - p_phys_r[None, :]
            dot_product_mat = np.sum(vec_grasp_mat * vec_base_direction, axis=2)
            crossing_mask = dot_product_mat < 0
            score_matrix[crossing_mask] = -np.inf

            # 4. 获取最优解
            max_idx_flat = np.argmax(score_matrix)
            max_score = score_matrix.flat[max_idx_flat]
            
            if max_score > best_total_score and max_score > -1e10:
                best_total_score = max_score
                
                # 还原回原始点云索引
                i_l, i_r = np.unravel_index(max_idx_flat, score_matrix.shape)
                final_l = curr_l_candidates[i_l]
                final_r = curr_r_candidates[i_r]
                
                best_pair = (final_l, final_r)
                canonical_dist = dist_matrix[i_l, i_r]
                
                best_debug_info = edict({
                    "topo_pair": (node_a_name, node_b_name),
                    "score": float(best_total_score),
                    "canonical_dist": float(dist_matrix[i_l, i_r]),
                    "phys_dist": float(dist_phys_mat[i_l, i_r]),
                    "dist_to_left_base": float(dist_to_left_base[i_l]),
                    "dist_to_right_base": float(dist_to_right_base[i_r]),
                })
        
        if best_pair[0] is not None:
            logger.info(
                f"[TopologicalGrasper] Selected edge {best_debug_info.topo_pair}, "
                f"score={best_debug_info.score:.4f}, "
                f"dist_to_base: L={best_debug_info.dist_to_left_base:.3f}m, "
                f"R={best_debug_info.dist_to_right_base:.3f}m"
            )

            # ---- Optional inward jitter post-processing ----
            # Replace each selected vertex with a nearby vertex (broadens grasp
            # coverage so the policy sees more than just the 4 fixed corners).
            if self._jitter_enabled:
                jittered_l = self._jitter_grasp_index(best_pair[0], cloth_pos)
                jittered_r = self._jitter_grasp_index(best_pair[1], cloth_pos)
                if jittered_l != best_pair[0] or jittered_r != best_pair[1]:
                    best_debug_info["jittered_from"] = (int(best_pair[0]), int(best_pair[1]))
                    best_pair = (jittered_l, jittered_r)
                    best_debug_info["jittered_to"] = (int(jittered_l), int(jittered_r))

        if visualize:
            self.visualize_grasp_analysis(
                cloth_pos=cloth_pos,
                vis_indices=vis_indices,
                left_vis_reach_indices=left_vis_reachable_indices,
                right_vis_reach_indices=right_vis_reachable_indices,
                selected_left_idx=best_pair[0],
                selected_right_idx=best_pair[1],
                debug_info=best_debug_info,
                base_left=base_left,
                base_right=base_right,
                workspace_radius=workspace_radius,
                show_core_regions=True,
                show_topology=True,
                show_workspace=True,
            )


        return best_pair[0], best_pair[1], canonical_dist, best_debug_info
    
    def visualize_grasp_analysis(
        self,
        cloth_pos: np.ndarray,
        vis_indices: np.ndarray,
        left_vis_reach_indices: np.ndarray,
        right_vis_reach_indices: np.ndarray,
        selected_left_idx: Optional[int] = None,
        selected_right_idx: Optional[int] = None,
        debug_info: Optional[Dict] = None,
        base_left: Optional[np.ndarray] = None,
        base_right: Optional[np.ndarray] = None,
        workspace_radius: float = 0.75,
        show_core_regions: bool = True,
        show_topology: bool = True,
        show_workspace: bool = False,  # 工作空间圈可选
        point_size: float = 3.0
    ):
        """
        统一可视化：可达性分析 + 抓取点选择
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Grasp Analysis (All-in-One)", width=1920, height=1080)
        
        # ============ 1. 布料顶点按可见性和可达性分类 ============
        vis_set = set(vis_indices)
        left_reach_set = set(left_vis_reach_indices)
        right_reach_set = set(right_vis_reach_indices)
        
        # 不可见点（深灰色，透明度低）
        all_indices = set(range(len(cloth_pos)))
        invisible_indices = list(all_indices - vis_set)
        if len(invisible_indices) > 0:
            pcd_invisible = o3d.geometry.PointCloud()
            pcd_invisible.points = o3d.utility.Vector3dVector(cloth_pos[invisible_indices])
            pcd_invisible.paint_uniform_color([0.3, 0.3, 0.3])
            vis.add_geometry(pcd_invisible)
        
        # 可见但不可达（黄色）
        vis_not_reach = list(vis_set - left_reach_set - right_reach_set)
        if len(vis_not_reach) > 0:
            pcd_vis_only = o3d.geometry.PointCloud()
            pcd_vis_only.points = o3d.utility.Vector3dVector(cloth_pos[vis_not_reach])
            pcd_vis_only.paint_uniform_color([0.8, 0.6, 0.0])
            vis.add_geometry(pcd_vis_only)
        
        # 左臂专属可达（蓝色渐变）
        left_only = list(left_reach_set - right_reach_set)
        if len(left_only) > 0:
            pcd_left = o3d.geometry.PointCloud()
            pcd_left.points = o3d.utility.Vector3dVector(cloth_pos[left_only])
            colors = self._weight_to_color(self.vertex_weights[left_only], colormap='Blues')
            pcd_left.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd_left)
        
        # 右臂专属可达（绿色渐变）
        right_only = list(right_reach_set - left_reach_set)
        if len(right_only) > 0:
            pcd_right = o3d.geometry.PointCloud()
            pcd_right.points = o3d.utility.Vector3dVector(cloth_pos[right_only])
            colors = self._weight_to_color(self.vertex_weights[right_only], colormap='Greens')
            pcd_right.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd_right)
        
        # 双臂都可达（紫色渐变）
        both_reach = list(left_reach_set & right_reach_set)
        if len(both_reach) > 0:
            pcd_both = o3d.geometry.PointCloud()
            pcd_both.points = o3d.utility.Vector3dVector(cloth_pos[both_reach])
            colors = self._weight_to_color(self.vertex_weights[both_reach], colormap='Purple')
            pcd_both.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd_both)
        
        if show_core_regions:
            for node_name in self.node_names:
                cfg = self.nodes_config[node_name]
                center_idx = cfg['index']
                core_radius = cfg.get('core_radius', 0.05)
                center_pos = cloth_pos[center_idx]
                
                # 只画3个正交的圆圈（XY, XZ, YZ平面）
                # 1. XY平面圆（赤道）
                angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
                points_xy = np.column_stack([
                    center_pos[0] + core_radius * np.cos(angles),
                    center_pos[1] + core_radius * np.sin(angles),
                    np.full(48, center_pos[2])
                ])
                lines_xy = [[i, (i+1) % 48] for i in range(48)]
                circle_xy = o3d.geometry.LineSet()
                circle_xy.points = o3d.utility.Vector3dVector(points_xy)
                circle_xy.lines = o3d.utility.Vector2iVector(lines_xy)
                circle_xy.paint_uniform_color([1.0, 0.8, 0.5])  # 浅黄橙色
                vis.add_geometry(circle_xy)
                
                # 2. XZ平面圆（经线1）
                points_xz = np.column_stack([
                    center_pos[0] + core_radius * np.cos(angles),
                    np.full(48, center_pos[1]),
                    center_pos[2] + core_radius * np.sin(angles)
                ])
                circle_xz = o3d.geometry.LineSet()
                circle_xz.points = o3d.utility.Vector3dVector(points_xz)
                circle_xz.lines = o3d.utility.Vector2iVector(lines_xy)
                circle_xz.paint_uniform_color([1.0, 0.8, 0.5])
                vis.add_geometry(circle_xz)
                
                # 3. YZ平面圆（经线2）
                points_yz = np.column_stack([
                    np.full(48, center_pos[0]),
                    center_pos[1] + core_radius * np.cos(angles),
                    center_pos[2] + core_radius * np.sin(angles)
                ])
                circle_yz = o3d.geometry.LineSet()
                circle_yz.points = o3d.utility.Vector3dVector(points_yz)
                circle_yz.lines = o3d.utility.Vector2iVector(lines_xy)
                circle_yz.paint_uniform_color([1.0, 0.8, 0.5])
                vis.add_geometry(circle_yz)
                
                # 中心十字标记（小一点，更精致）
                cross_size = 0.006
                colors_xyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                for axis, color in zip([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])], colors_xyz):
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([
                        center_pos - axis * cross_size,
                        center_pos + axis * cross_size
                    ])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.paint_uniform_color(color)
                    vis.add_geometry(line)
        
        # ============ 3. 拓扑连接图（虚线） ============
        if show_topology:
            for node_a_name, node_b_name in self.adjacency_graph:
                idx_a = self.nodes_config[node_a_name]['index']
                idx_b = self.nodes_config[node_b_name]['index']
                
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([
                    cloth_pos[idx_a],
                    cloth_pos[idx_b]
                ])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.paint_uniform_color([0.5, 0.5, 0.5])
                vis.add_geometry(line)
        
        # ============ 4. 最终选中的抓取点（醒目小球 + 箭头标注） ============
        if selected_left_idx is not None:
            # 小红球（改小，避免遮挡）
            sphere_left = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            sphere_left.translate(cloth_pos[selected_left_idx])
            # sphere_left.paint_uniform_color([1.0, 0.0, 0.0])
            sphere_left.paint_uniform_color([0.0, 0.4, 1.0]) 
            sphere_left.compute_vertex_normals()
            vis.add_geometry(sphere_left)
            
            # 箭头标注（向上）
            arrow_left = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.002,
                cone_radius=0.005,
                cylinder_height=0.04,
                cone_height=0.015
            )
            # 旋转箭头指向上方
            arrow_left.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]),
                center=[0, 0, 0]
            )
            arrow_left.translate(cloth_pos[selected_left_idx] + np.array([0, 0, 0.06]))
            # arrow_left.paint_uniform_color([1.0, 0.0, 0.0])
            arrow_left.paint_uniform_color([0.0, 0.4, 1.0]) 
            vis.add_geometry(arrow_left)
        
        if selected_right_idx is not None:
            sphere_right = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            sphere_right.translate(cloth_pos[selected_right_idx])
            # sphere_right.paint_uniform_color([1.0, 0.0, 0.0])
            sphere_right.paint_uniform_color([0.0, 1.0, 0.4])
            sphere_right.compute_vertex_normals()
            vis.add_geometry(sphere_right)
            
            arrow_right = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.002,
                cone_radius=0.005,
                cylinder_height=0.04,
                cone_height=0.015
            )
            arrow_right.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]),
                center=[0, 0, 0]
            )
            arrow_right.translate(cloth_pos[selected_right_idx] + np.array([0, 0, 0.06]))
            # arrow_right.paint_uniform_color([1.0, 0.0, 0.0])
            arrow_right.paint_uniform_color([0.0, 1.0, 0.4]) 
            vis.add_geometry(arrow_right)
        
        # 抓取对连接线（粗橙色线）
        if selected_left_idx is not None and selected_right_idx is not None:
            line_grasp = o3d.geometry.LineSet()
            line_grasp.points = o3d.utility.Vector3dVector([
                cloth_pos[selected_left_idx],
                cloth_pos[selected_right_idx]
            ])
            line_grasp.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_grasp.paint_uniform_color([1.0, 0.5, 0.0])
            vis.add_geometry(line_grasp)
        
        # ============ 5. 机械臂基座（可选） ============
        if show_workspace and base_left is not None:
            sphere_base_left = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere_base_left.translate(base_left[:3])
            sphere_base_left.paint_uniform_color([0.0, 0.3, 0.8])
            vis.add_geometry(sphere_base_left)
            
            workspace_left = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=workspace_radius, tube_radius=0.003
            )
            workspace_left.translate(base_left[:3])
            workspace_left.paint_uniform_color([0.2, 0.5, 1.0])
            workspace_left.compute_vertex_normals()
            vis.add_geometry(workspace_left)
        
        if show_workspace and base_right is not None:
            sphere_base_right = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere_base_right.translate(base_right[:3])
            sphere_base_right.paint_uniform_color([0.0, 0.8, 0.3])
            vis.add_geometry(sphere_base_right)
            
            workspace_right = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=workspace_radius, tube_radius=0.003
            )
            workspace_right.translate(base_right[:3])
            workspace_right.paint_uniform_color([0.2, 1.0, 0.5])
            workspace_right.compute_vertex_normals()
            vis.add_geometry(workspace_right)
        
        # ============ 6. 坐标系 ============
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(coord_frame)
        
        # ============ 7. 渲染设置 ============
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.mesh_show_back_face = True
        opt.line_width = 2.0
        
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(cloth_pos.mean(axis=0))
        ctr.set_up([0, 1, 0])
        
        # ============ 8. 详细日志 ============
        # logger.info("=== Unified Grasp Analysis Visualization ===")
        # logger.info(f"  Dark Gray:  Invisible vertices ({len(invisible_indices)})")
        # logger.info(f"  Yellow:     Visible but unreachable ({len(vis_not_reach)})")
        # logger.info(f"  Blue:       Left arm only ({len(left_only)})")
        # logger.info(f"  Green:      Right arm only ({len(right_only)})")
        # logger.info(f"  Purple:     Both arms reachable ({len(both_reach)})")
        # logger.info(f"  Orange wireframe: Core regions (radius={list(self.nodes_config.values())[0].get('core_radius', 0.05)}m)")
        # logger.info(f"  Red arrows: Selected grasp points")
        # logger.info(f"  Orange line: Final grasp pair")
        
        if debug_info:
            logger.info(f"\n=== Selection Details ===")
            logger.info(f"  Topology: {debug_info.get('topo_pair', 'N/A')}")
            logger.info(f"  Score:    {debug_info.get('score', 0):.2f}")
            logger.info(f"  Distance: {debug_info.get('canonical_dist', 0):.4f}m")
            logger.info(f"  Left idx: {selected_left_idx}, Right idx: {selected_right_idx}")
            if selected_left_idx is not None:
                logger.info(f"  Left weight: {self.vertex_weights[selected_left_idx]:.2f}")
            if selected_right_idx is not None:
                logger.info(f"  Right weight: {self.vertex_weights[selected_right_idx]:.2f}")
        
        vis.run()
        vis.destroy_window()


    def _weight_to_color(self, weights: np.ndarray, colormap: str = 'Blues') -> np.ndarray:
        """改进权重到颜色映射，增加紫色支持"""
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            normalized = (weights - w_min) / (w_max - w_min)
        else:
            normalized = np.ones_like(weights) * 0.5
        
        colors = np.zeros((len(weights), 3))
        if colormap == 'Blues':
            colors[:, 2] = 0.3 + 0.7 * normalized
            colors[:, 0] = 0.05
            colors[:, 1] = 0.1 + 0.3 * normalized
        elif colormap == 'Greens':
            colors[:, 1] = 0.3 + 0.7 * normalized
            colors[:, 0] = 0.05
            colors[:, 2] = 0.1 + 0.3 * normalized
        elif colormap == 'Purple':
            colors[:, 0] = 0.4 + 0.6 * normalized  # R
            colors[:, 1] = 0.1
            colors[:, 2] = 0.4 + 0.6 * normalized  # B
        
        return colors
