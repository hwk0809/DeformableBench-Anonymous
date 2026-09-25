import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List,Union

def visualize_occlusion_check(vertices_idx: int, 
                            cloth_pos: np.ndarray,
                            safe_neighbor_map: dict,
                            gripper_radius: float = 0.02,
                            check_height: float = 0.10,
                            save_path: str = None):
    """
    可视化遮挡检测结果
    
    Args:
        vertices_idx: 要检查的顶点索引
        cloth_pos: 布料所有顶点位置
        gripper_radius: 夹爪半径
        check_height: 检测圆柱体高度
        save_path: 保存路径，如果为None则显示图像
    """

        
    target_pos = cloth_pos[vertices_idx]
    
    # 1. 空间过滤
    diff = cloth_pos - target_pos
    dist_xy = np.linalg.norm(diff[:, :2], axis=1)
    dist_z = diff[:, 2]
    
    cylinder_mask = (dist_xy < gripper_radius) & (dist_z > 0.005) & (dist_z < check_height)
    potential_blockers = np.where(cylinder_mask)[0]
    
    # 2. 拓扑剔除
    safe_neighbors = safe_neighbor_map[vertices_idx]
    is_safe_neighbor = np.isin(potential_blockers, safe_neighbors)
    real_blockers = potential_blockers[~is_safe_neighbor]
    safe_blockers = potential_blockers[is_safe_neighbor]
    
    # 3. 绘制3D可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1: 3D全局视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(cloth_pos[:, 0], cloth_pos[:, 1], cloth_pos[:, 2], 
                c='lightgray', s=1, alpha=0.3, label='All vertices')
    ax1.scatter(*target_pos, c='green', s=100, marker='*', 
                label=f'Target vertex {vertices_idx}', zorder=10)
    
    if len(safe_blockers) > 0:
        ax1.scatter(cloth_pos[safe_blockers, 0], 
                    cloth_pos[safe_blockers, 1], 
                    cloth_pos[safe_blockers, 2],
                    c='yellow', s=30, alpha=0.6, label='Safe neighbors')
    
    if len(real_blockers) > 0:
        ax1.scatter(cloth_pos[real_blockers, 0], 
                    cloth_pos[real_blockers, 1], 
                    cloth_pos[real_blockers, 2],
                    c='red', s=50, marker='x', label='Real blockers', zorder=9)
    
    # 绘制检测圆柱体
    theta = np.linspace(0, 2*np.pi, 30)
    z_cyl = np.linspace(target_pos[2], target_pos[2] + check_height, 10)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = target_pos[0] + gripper_radius * np.cos(theta_grid)
    y_cyl = target_pos[1] + gripper_radius * np.sin(theta_grid)
    ax1.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.2, color='blue')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View - Occlusion Check')
    ax1.legend()
    
    # 子图2: XY平面俯视图
    ax2 = fig.add_subplot(132)
    ax2.scatter(cloth_pos[:, 0], cloth_pos[:, 1], 
                c='lightgray', s=5, alpha=0.3)
    ax2.scatter(*target_pos[:2], c='green', s=200, marker='*', 
                label=f'Target {vertices_idx}', zorder=10)
    
    # 绘制圆柱体投影圆
    circle = plt.Circle((target_pos[0], target_pos[1]), gripper_radius, 
                        fill=False, color='blue', linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    
    if len(safe_blockers) > 0:
        ax2.scatter(cloth_pos[safe_blockers, 0], 
                    cloth_pos[safe_blockers, 1],
                    c='yellow', s=50, alpha=0.6, label='Safe neighbors')
    
    if len(real_blockers) > 0:
        ax2.scatter(cloth_pos[real_blockers, 0], 
                    cloth_pos[real_blockers, 1],
                    c='red', s=80, marker='x', label='Real blockers', zorder=9)
        # 标注第一个真实遮挡物
        first_blocker_pos = cloth_pos[real_blockers[0]]
        ax2.annotate(f'Blocker {real_blockers[0]}', 
                    xy=(first_blocker_pos[0], first_blocker_pos[1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane View (Top Down)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 高度分布直方图
    ax3 = fig.add_subplot(133)
    if len(potential_blockers) > 0:
        heights = cloth_pos[potential_blockers, 2] - target_pos[2]
        ax3.hist(heights, bins=20, alpha=0.5, color='gray', label='All in cylinder')
        
        if len(safe_blockers) > 0:
            safe_heights = cloth_pos[safe_blockers, 2] - target_pos[2]
            ax3.hist(safe_heights, bins=20, alpha=0.6, color='yellow', label='Safe neighbors')
        
        if len(real_blockers) > 0:
            real_heights = cloth_pos[real_blockers, 2] - target_pos[2]
            ax3.hist(real_heights, bins=20, alpha=0.8, color='red', label='Real blockers')
            ax3.axvline(x=np.min(real_heights), color='red', linestyle='--', 
                        label=f'Min blocker height: {np.min(real_heights):.4f}m')
    
    ax3.axhline(y=0, color='green', linestyle='-', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Height above target (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Height Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Occlusion Check for Vertex {vertices_idx}\n'
                f'Potential: {len(potential_blockers)}, '
                f'Safe: {len(safe_blockers)}, '
                f'Real Blockers: {len(real_blockers)}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Occlusion visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印详细信息
    logger.info(f"\n{'='*60}")
    logger.info(f"Occlusion Check Results for Vertex {vertices_idx}")
    logger.info(f"{'='*60}")
    logger.info(f"Target position: {target_pos}")
    logger.info(f"Cylinder radius: {gripper_radius}m, Height: {check_height}m")
    logger.info(f"Potential blockers in cylinder: {len(potential_blockers)}")
    logger.info(f"Safe neighbors (topology): {len(safe_blockers)}")
    logger.info(f"Real blockers: {len(real_blockers)}")
    
    if len(real_blockers) > 0:
        logger.warning(f"⚠️  Vertex IS OCCLUDED!")
        logger.info(f"Real blocker indices: {real_blockers[:5]}...")  # 只显示前5个
        min_blocker_idx = real_blockers[np.argmin(cloth_pos[real_blockers, 2])]
        min_blocker_height = cloth_pos[min_blocker_idx, 2] - target_pos[2]
        logger.info(f"Closest blocker: vertex {min_blocker_idx} at {min_blocker_height:.4f}m above target")
    else:
        logger.success(f"✓ Vertex is NOT occluded - safe to grasp!")
    logger.info(f"{'='*60}\n")




def visualize_cloth_topology_o3d(cloth_pos: np.ndarray,
                                 cloth_corners_vertices: Union[List[int], np.ndarray] = None,
                                 cloth_edges_vertices: Union[List[List[int]], np.ndarray] = None,
                                 cloth_edge_centers_vertices: Union[List[int], np.ndarray] = None,
                                 save_path: str = None):
    """
    使用 Open3D 可视化布料拓扑结构（角点、边、边中点）
    
    Args:
        cloth_pos: 布料所有顶点位置 (N, 3)
        save_path: 保存路径，如果为None则交互显示
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("Open3D not installed. Install with: pip install open3d")
        return
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloth_pos)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色 - 所有点
    
    # 准备几何体列表
    geometries = [pcd]
    
    # 1. 标记四个角点（红色大球）
    if cloth_corners_vertices is not None:
        for i, corner_idx in enumerate(cloth_corners_vertices):
            corner_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            corner_sphere.translate(cloth_pos[corner_idx])
            corner_sphere.paint_uniform_color([1, 0, 0])  # 红色
            geometries.append(corner_sphere)
            
            # 添加文字标签（通过小坐标轴表示）
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.03, origin=cloth_pos[corner_idx]
            )
            geometries.append(coord_frame)
            
            logger.info(f"Corner {i}: vertex {corner_idx} at {cloth_pos[corner_idx]}")
    
    # 2. 标记边中点（蓝色中球）
    if cloth_edge_centers_vertices is not None:
        for i, center_idx in enumerate(cloth_edge_centers_vertices):
            center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            center_sphere.translate(cloth_pos[center_idx])
            center_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
            geometries.append(center_sphere)
            
            logger.info(f"Edge {i} center: vertex {center_idx} at {cloth_pos[center_idx]}")
    
    # 3. 绘制边（绿色线条）
    if cloth_edges_vertices is not None:
        for i, edge_vertices in enumerate(cloth_edges_vertices):
            # 提取边上所有点的坐标
            edge_points = cloth_pos[edge_vertices]
            
            # 创建线段
            lines = [[j, j+1] for j in range(len(edge_vertices)-1)]
            colors = [[0, 1, 0] for _ in lines]  # 绿色
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(edge_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
            
            logger.info(f"Edge {i}: {len(edge_vertices)} vertices from corner {i} to corner {(i+1)%4}")
    
    # 4. 连接相邻角点（黄色虚线，验证拓扑顺序）
    if cloth_corners_vertices is not None:
        corner_positions = cloth_pos[cloth_corners_vertices]
        corner_lines = []
        for i in range(4):
            corner_lines.append([i, (i+1)%4])
        
        corner_line_set = o3d.geometry.LineSet()
        corner_line_set.points = o3d.utility.Vector3dVector(corner_positions)
        corner_line_set.lines = o3d.utility.Vector2iVector(corner_lines)
        corner_line_set.colors = o3d.utility.Vector3dVector([[1, 1, 0] for _ in corner_lines])  # 黄色
        geometries.append(corner_line_set)
    
    # 添加坐标系（世界坐标）
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    geometries.append(world_frame)
    
    # 可视化
    logger.info("\n" + "="*60)
    logger.info("Open3D Visualization Controls:")
    logger.info("  - Mouse Left: Rotate")
    logger.info("  - Mouse Right: Pan")
    logger.info("  - Mouse Wheel: Zoom")
    logger.info("  - Press 'H': Show help")
    logger.info("  - Press 'Q' or ESC: Close window")
    logger.info("="*60)
    logger.info("\nColor Legend:")
    logger.info("  🔴 Red Spheres: Corner vertices")
    logger.info("  🔵 Blue Spheres: Edge center vertices")
    logger.info("  🟢 Green Lines: Edges (connecting corners)")
    logger.info("  🟡 Yellow Lines: Corner connections (topology order)")
    logger.info("  ⬜ Gray Points: All cloth vertices")
    logger.info("="*60 + "\n")
    
    if save_path:
        # 保存为PLY文件
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geom in geometries:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        logger.success(f"Topology visualization saved to {save_path}")
    else:
        # 交互显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Cloth Topology Visualization",
            width=1280,
            height=720,
            left=50,
            top=50
        )