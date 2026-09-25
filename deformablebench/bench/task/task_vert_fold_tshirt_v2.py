import os
import sys
import numpy as np
from typing import Optional, Any, Generator
from omegaconf import DictConfig
from loguru import logger
from scipy.spatial.transform import Rotation as R

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deformablebench.bench.task.task_base_fold import BaseFoldTask
from deformablebench.bench.common.types import ArmSide, ExecutionStep
from deformablebench.bench.robot.planning.planner_manager import PlannerManager


class TShirtFoldTask(BaseFoldTask):
    """
    T恤折叠任务 (V2 - 无随机化, 平整初始状态)

    策略:
      Stage 1:
      Stage 2:
    """

    # 高度常量
    SAFE_HEIGHT = 0.15  # waypoint安全高度
    TABLE_HEIGHT = 0.01  # 桌面/布料表面z
    GRASP_Z_OFFSET = -0.01  # 抓取时夹爪略低于布料
    GRASP_Z_OFFSET_STEP = -0.005  # 每次重试额外下压量
    OPEN_GRIPPER = 0.9
    OPEN_SMALL_GRIPPER = 0.9
    CLOSE_GRIPPER = 0.0

    # Stage 2, 3 前折参数 (Boss 可调)
    FORWARD_FOLD_LIFT_HEIGHT = 0.25  # 前折时先抬多高再观测
    FOLD_DEST_Z_OFFSET = 0.05  # 折叠目的地比桌面高多少
    FORCED_GRASP_Z = 0.001  # 指定 grasp pose 绝对 z，绕过 task 内部 z_offset/clamp

    def __init__(
        self,
        env,
        planner_manager: PlannerManager,
        collector=None,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__(
            env=env,
            keypoints=None,
            planner_manager=planner_manager,
            collector=collector,
            cfg=cfg,
        )
        self.stage_1_success = False
        self.stage_2_success = False
        self.stage_3_success = False

        # fold_type 从 cfg.task_params 读取，缺省走 v2 (sleeves -> sides -> vertical)
        tp: dict = {}
        if cfg is not None and hasattr(cfg, "get"):
            tp = dict(cfg.get("task_params", {}) or {})
        self.fold_type = str(tp.get("fold_type", "v2"))

        self.task_meta = {
            "episode_id": 0,
            "fold_type": self.fold_type,
            "description": "Fold the green T-shirt in a dual-arm coordinated manner",
            "strategy": {
                "stage1": "Simultaneously grasp the upper sleeves with both arms, fold inward along a continuous arc to the torso",
                "stage2": "Grasp the shoulders and fold forward along a continuous arc (fold upper body to the hem)",
            },
        }

    @staticmethod
    def _set_pose_z(pose: np.ndarray, z: float) -> np.ndarray:
        pose = pose.copy()
        pose[2] = z
        return pose

    @staticmethod
    def _rotate_pose_about_world_z(pose: np.ndarray, angle_deg: float) -> np.ndarray:
        pose = pose.copy()
        quat_wxyz = np.asarray(pose[3:7], dtype=np.float64)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot_current = R.from_quat(quat_xyzw)
        rot_delta = R.from_euler("z", angle_deg, degrees=True)
        quat_new_xyzw = (rot_delta * rot_current).as_quat()
        pose[3:7] = np.array(
            [quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]]
        )
        return pose

    # ========================================
    # Main Episode
    # ========================================
    def plan_episode(self, episode_id: int) -> Generator[Any, None, None]:
        """
        裤子折叠完整策略: Stage 1 (左右折叠) → Stage 2 (上下折叠)
        """
        logger.info(f"[PantsFoldTask] Starting episode {episode_id}")
        self.task_meta["episode_id"] = episode_id

        # ========== Stage 1: 袖子折叠 ==========
        with self.recording("stage1_sleeve_fold"):
            success = yield from self._bimanual_sleeve_fold()
            if not success:
                logger.error("[TShirtFoldTask] Stage 1 (sleeve fold) failed")
                raise IndexError("[TShirtFoldTask] Stage 1 (sleeve fold) failed")
                return
            self.stage_1_success = True
            logger.success("[TShirtFoldTask] Stage 1 (sleeve fold) completed")

        if self.fold_type == "v2":
            # ========== Stage 2: 左右折叠 ==========
            with self.recording("stage2_side_fold"):
                success = yield from self._stage_side_fold()
                if not success:
                    logger.error("[TShirtFoldTask] Stage 2 (side fold) failed")
                    raise IndexError("[TShirtFoldTask] Stage 2 (side fold) failed")
                    return
                self.stage_2_success = True
                logger.success("[TShirtFoldTask] Stage 2 (side fold) completed")

            # ========== Stage 3: 前折 ==========
            with self.recording("stage3_vertical_fold"):
                success = yield from self._stage_bimanual_vertical_fold()
                if not success:
                    logger.error("[TShirtFoldTask] Stage 3 (vertical fold) failed")
                    raise IndexError("[TShirtFoldTask] Stage 3 (vertical fold) failed")
                    return
                self.stage_3_success = True
                logger.success("[TShirtFoldTask] Stage 3 (vertical fold) completed")

            logger.success("[TShirtFoldTask] All stages completed!")
        else:
            with self.recording("stage2_forward_fold"):
                success = yield from self._stage_forward_fold()
                if not success:
                    logger.error("[TShirtFoldTask] Stage 2 (forward fold) failed")
                    raise IndexError("[TShirtFoldTask] Stage 2 (forward fold) failed")
                    return
                self.stage_2_success = True
                self.stage_3_success = True
                logger.success("[TShirtFoldTask] Stage 2 (forward fold) completed!")

        self.task_success = True

    # ========================================
    # Stage 1: 袖子折叠
    # ========================================
    def _bimanual_sleeve_fold(self) -> Generator[ExecutionStep, None, bool]:
        """
        抓取双袖上部 → 连续弧线内收 → 放到躯干对称位置

        运动轨迹（抓住后一次waypoint规划完成）:
          当前位置(test_lift后) → 上方内侧(shoulder正上方) → 折叠目标(Y轴镜像)
        """

        # Phase A: 用 robust_bimanual_grasp 抓取袖子
        # 注意: 衣服是反放的, 左臂(Y+侧)抓右袖, 右臂(Y-侧)抓左袖
        def sleeve_targets(obs, specific_indices=None, z_offset=None):
            pose_l, pose_r, l_idx, r_idx = self._make_keypoint_grasp_targets(
                obs,
                "right_sleeve_outer",
                "left_sleeve_outer",
                specific_indices,
                z_offset=z_offset,
            )
            # pose_l = self._set_pose_z(pose_l, self.FORCED_GRASP_Z)
            # pose_r = self._set_pose_z(pose_r, self.FORCED_GRASP_Z)
            pose_l = pose_l.copy()
            pose_l[1] -= 0.003
            pose_l[2] -= 0.0025
            pose_r = pose_r.copy()
            pose_r[1] += 0.003
            pose_r[2] -= 0.0025

            return pose_l, pose_r, l_idx, r_idx

        success = yield from self.skills.robust_bimanual_grasp(
            close_gripper_value=self.grasp_val,  # T64: 读 yaml grasp_close_val(生产 kinematic yaml 显式 0.015,逐位不变)
            get_targets_func=sleeve_targets,
            max_retries=3,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            retry_inward_step=self.GRASP_XY_OFFSET_STEP,
            grasp_settle_time=self.grasp_settle_time,
            test_lift_height=self.test_lift_height,
            y_retract_offset=0.02,
            velocity_scale=0.4,
            ground_check=False,
        )
        if not success:
            return False

        # Phase B: 连续弧线折叠
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        cloth_pos = obs.pcd_cloth_truth.copy()

        # 读取关键点位置 (左臂抓的是R_sleeve, 右臂抓的是L_sleeve)
        l_arm_sleeve = self._get_keypoint_pos(
            cloth_pos, "right_sleeve_outer"
        )  # 左臂抓的袖子
        r_arm_sleeve = self._get_keypoint_pos(
            cloth_pos, "left_sleeve_outer"
        )  # 右臂抓的袖子
        l_arm_shoulder = self._get_keypoint_pos(cloth_pos, "right_shoulder")  # 左臂侧的肩
        r_arm_shoulder = self._get_keypoint_pos(cloth_pos, "left_shoulder")  # 右臂侧的肩

        # Waypoint 1: 上方内侧 (lift + 内收到shoulder正上方)
        safe_height = l_arm_sleeve[0] - l_arm_shoulder[0] + 0.02  # 基于袖肩水平距离计算安全高度，确保内收过程中不碰桌面
        # l_arm_shoulder[0] += 0.1
        # r_arm_shoulder[0] += 0.1
        logger.debug(f"[Sleeve Fold] Safe height: {safe_height}")
        l_upper_inward_pos = np.array([l_arm_sleeve[0], l_arm_shoulder[1], safe_height])
        r_upper_inward_pos = np.array([r_arm_sleeve[0], r_arm_shoulder[1], safe_height])
        quat_l_inward = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.left_base, l_upper_inward_pos
        )
        quat_r_inward = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.right_base, r_upper_inward_pos
        )
        l_upper_inward = np.concatenate([l_upper_inward_pos, quat_l_inward])
        r_upper_inward = np.concatenate([r_upper_inward_pos, quat_r_inward])

        # Waypoint 2: 折叠目标 (Y轴镜像: fold_y = 2*shoulder_y - sleeve_y)
        l_fold_y = 2 * l_arm_shoulder[1] - l_arm_sleeve[1] + 0.035
        r_fold_y = 2 * r_arm_shoulder[1] - r_arm_sleeve[1] - 0.035
        l_fold_x = l_arm_sleeve[0]
        r_fold_x = r_arm_sleeve[0]
        l_fold_target_pos = np.array([l_fold_x, l_fold_y, self.TABLE_HEIGHT])
        r_fold_target_pos = np.array([r_fold_x, r_fold_y, self.TABLE_HEIGHT])
        quat_l_fold = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.left_base, l_fold_target_pos
        )
        quat_r_fold = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.right_base, r_fold_target_pos
        )
        l_fold_target = np.concatenate([l_fold_target_pos, quat_l_fold])
        r_fold_target = np.concatenate([r_fold_target_pos, quat_r_fold])

        logger.info(
            f"[Sleeve Fold] L: upper_inward={l_upper_inward}, fold_target={l_fold_target}"
        )
        logger.info(
            f"[Sleeve Fold] R: upper_inward={r_upper_inward}, fold_target={r_fold_target}"
        )

        # 一次waypoint规划: 连续弧线 (不停顿)
        plan_fold = self.skills.move_dual_arm_to(
            left_target_pose=[l_upper_inward, l_fold_target],
            right_target_pose=[r_upper_inward, r_fold_target],
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
            # interpolation_mode=["joint", "joint"],
            # steps_per_segment=10
            ee_link="tcp_link_bottom",
            velocity_scale=0.8,
        )
        yield ExecutionStep(
            plan=plan_fold, settle_time=0.1, description="sleeve_fold_arc"
        )

        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        l_lift_pos = np.array([l_ee[0]-0.05, l_ee[1] + 0.08, 0.3])
        r_lift_pos = np.array([r_ee[0]-0.05, r_ee[1] - 0.08, 0.3])

        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=l_lift_pos,
            right_target_pose=r_lift_pos,
            left_gripper=self.OPEN_GRIPPER,
            right_gripper=self.OPEN_GRIPPER,
            pose_constraint=[1, 1, 0.8, 0, 0, 0],
        )
        yield ExecutionStep(
            plan=plan_retract, settle_time=0.3, description="sleeve_release_retract"
        )

        return True

    def _stage_side_fold(self) -> Generator[Any, None, bool]:
        """
        Stage 2: T恤左右折叠"""

        def pants_targets(obs, specific_indices=None, z_offset=None):
            return self._make_keypoint_grasp_targets(
                obs,
                left_key="left_shoulder",
                right_key="left_hem",
                specific_indices=specific_indices,
                z_offset=z_offset,
            )

        success = yield from self.skills.robust_bimanual_grasp(
            close_gripper_value=self.grasp_val,  # T64: 读 yaml grasp_close_val(生产 kinematic yaml 显式 0.015,逐位不变)
            get_targets_func=pants_targets,
            max_retries=3,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            retry_inward_step=self.GRASP_XY_OFFSET_STEP,
            grasp_settle_time=self.grasp_settle_time,
            test_lift_height=self.test_lift_height,
            y_retract_offset=0.02,
        )
        if not success:
            return False

        obs = self.reset_interface_state()  # 抓取成功后重置状态以获取新观测
        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        l_lift_pos = np.array([l_ee[0], l_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])
        r_lift_pos = np.array([r_ee[0], r_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])

        plan_lift = self.skills.move_dual_arm_to(
            left_target_pose=l_lift_pos,
            right_target_pose=r_lift_pos,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(
            plan=plan_lift, settle_time=0.3, description="forward_fold_lift"
        )

        obs = self.reset_interface_state()  # 抓取成功后重置状态以获取新观测
        cloth_pos = obs.pcd_cloth_truth.copy()

        l_target_pos = self._get_keypoint_pos(cloth_pos, "right_shoulder")
        r_target_pos = self._get_keypoint_pos(cloth_pos, "right_hem")

        l_fold_target_pos = np.array([l_target_pos[0] - 0.005, l_target_pos[1], 0.02])
        r_fold_target_pos = np.array([r_target_pos[0] - 0.005, r_target_pos[1], 0.02])

        quat_l_fold = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.left_base, l_fold_target_pos
        )
        quat_r_fold = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.right_base, r_fold_target_pos
        )

        l_fold_target = np.concatenate([l_fold_target_pos, quat_l_fold])
        r_fold_target = np.concatenate([r_fold_target_pos, quat_r_fold])

        plan_fold = self.skills.move_dual_arm_to(
            left_target_pose=[l_fold_target],
            right_target_pose=[r_fold_target],
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0]],
            # interpolation_mode=["joint", "joint"],
            # steps_per_segment=10
            ee_link="tcp_link_bottom",
        )
        yield ExecutionStep(plan=plan_fold, settle_time=0.1, description="side fold")

        # 松手
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(
                self.OPEN_GRIPPER, self.OPEN_GRIPPER, duration=0.3
            ),
            settle_time=0.3,
            description="pants_release",
        )

        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=[0.2, 0.3, self.safe_height],
            right_target_pose=[0.2, -0.3, self.safe_height],
            pose_constraint=[1, 1, 0.8, 0, 0, 0],
        )
        yield ExecutionStep(
            plan=plan_retract, settle_time=0.3, description="fold_retract"
        )

        return True

    # ========================================
    # Stage 2: 前折
    # ========================================
    def _stage_forward_fold(self) -> Generator[ExecutionStep, None, bool]:
        """
        抓取双肩 → 连续弧线前折 → 放到下摆附近

        运动轨迹（抓住后一次waypoint规划完成）:
          当前位置(test_lift后) → 弧顶(neck正上方, z=0.25) → 下摆位置
        """

        # Phase A: 用 robust_bimanual_grasp 抓取肩部
        # 注意: 衣服是反放的, 左臂(Y+侧)抓右肩, 右臂(Y-侧)抓左肩
        def shoulder_targets(obs, specific_indices=None, z_offset=None):
            pose_l, pose_r, l_idx, r_idx =  self._make_keypoint_grasp_targets(
                obs, "right_shoulder", "left_shoulder", specific_indices, z_offset=z_offset
            )
            pose_l = pose_l.copy()
            pose_l[1] += 0.015
            pose_r = pose_r.copy()
            pose_r[1] -= 0.01
            pose_l = self._set_pose_z(pose_l, self.FORCED_GRASP_Z)
            pose_r = self._set_pose_z(pose_r, self.FORCED_GRASP_Z)
            pose_l = self._rotate_pose_about_world_z(pose_l, 35.0)
            pose_r = self._rotate_pose_about_world_z(pose_r, -35.0)
            return pose_l, pose_r, l_idx, r_idx
        


        success = yield from self.skills.robust_bimanual_grasp(
            close_gripper_value=self.grasp_val,  # T64: 读 yaml grasp_close_val(生产 kinematic yaml 显式 0.015,逐位不变)
            get_targets_func=shoulder_targets,
            max_retries=3,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            retry_inward_step=self.GRASP_XY_OFFSET_STEP,
            grasp_settle_time=self.grasp_settle_time,
            test_lift_height=self.test_lift_height,
            y_retract_offset=0.02,
            ground_check=False,
            pose_constraint=[1,1,1,1,0,1],
            pose_constraint_frame="local"
        )
        if not success:
            return False

        # Phase B: 先抬起来, 让布料自然下垂收缩
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        l_lift_pos = np.array([l_ee[0], l_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])
        r_lift_pos = np.array([r_ee[0], r_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])

        plan_lift = self.skills.move_dual_arm_to(
            left_target_pose=l_lift_pos,
            right_target_pose=r_lift_pos,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(
            plan=plan_lift, settle_time=0.3, description="forward_fold_lift"
        )

        # Phase C: 重新观测——布料被抬起后顶点位置已改变
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        cloth_pos = obs.pcd_cloth_truth

        # 用新观测重新获取关键点位置 (左臂抓的是right_shoulder, 右臂抓的是left_shoulder)
        l_arm_shoulder = self._get_keypoint_pos(cloth_pos, "right_shoulder")  # 左臂侧
        r_arm_shoulder = self._get_keypoint_pos(cloth_pos, "left_shoulder")  # 右臂侧
        l_arm_hem = self._get_keypoint_pos(cloth_pos, "right_hem")  # 左臂侧下摆
        r_arm_hem = self._get_keypoint_pos(cloth_pos, "left_hem")  # 右臂侧下摆

        logger.info(
            f"[Forward Fold] After lift re-observe: L_shoulder={l_arm_shoulder}, R_shoulder={r_arm_shoulder}"
        )
        logger.info(
            f"[Forward Fold] After lift re-observe: L_hem={l_arm_hem}, R_hem={r_arm_hem}"
        )

        # Phase D: 基于新观测计算弧线前折

        l_arm_hem[0] = l_arm_hem[0]
        r_arm_hem[0] = r_arm_hem[0]
        l_arm_hem[1] = l_arm_shoulder[1]
        r_arm_hem[1] = r_arm_shoulder[1]

        # Waypoint 2: 折叠目的地 (基于新观测的 hem 位置 + 可调Z偏移)
        fold_dest_z = self.TABLE_HEIGHT
        l_fold_dest_pos = np.array([l_arm_hem[0] - 0.03, l_arm_hem[1] + 0.04, fold_dest_z])
        r_fold_dest_pos = np.array([r_arm_hem[0] - 0.03, r_arm_hem[1] - 0.04, fold_dest_z])
        quat_l_dest = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.left_base, l_fold_dest_pos
        )
        quat_r_dest = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.right_base, r_fold_dest_pos
        )
        l_fold_dest = np.concatenate([l_fold_dest_pos, quat_l_dest])
        r_fold_dest = np.concatenate([r_fold_dest_pos, quat_r_dest])

        logger.info(f"[Forward Fold] L:  fold_dest={l_fold_dest}")
        logger.info(f"[Forward Fold] R: fold_dest={r_fold_dest}")

        # Waypoint规划: 从抬起位置 → 弧顶 → 折叠目的地
        plan_arc = self.skills.move_dual_arm_to(
            left_target_pose=[l_fold_dest],
            right_target_pose=[r_fold_dest],
            goal_type="waypoints",
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
        )
        yield ExecutionStep(
            plan=plan_arc, settle_time=0.1, description="forward_fold_arc"
        )

        # 回撤到上方，同时松开夹爪
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=[l_fold_dest[0], l_fold_dest[1], self.safe_height],
            right_target_pose=[r_fold_dest[0], r_fold_dest[1], self.safe_height],
            left_gripper=self.OPEN_SMALL_GRIPPER,
            right_gripper=self.OPEN_SMALL_GRIPPER,
            pose_constraint=[1, 1, 1, 0, 0, 0],
        )
        yield ExecutionStep(
            plan=plan_retract, settle_time=0.3, description="forward_release_retract"
        )

        return True

    def _stage_bimanual_vertical_fold(self) -> Generator[Any, None, bool]:
        """
        Stage 3: 前折升级版 (双臂协作, 上半部分折向下半部分)
        """

        # Phase A: 用 robust_bimanual_grasp 抓取肩部
        # 注意: 衣服是反放的, 左臂(Y+侧)抓右肩, 右臂(Y-侧)抓左肩
        def shoulder_targets(obs, specific_indices=None, z_offset=None):
            s_pos, c_pos, c_idx, s_idx = self._make_keypoint_grasp_targets(
                obs, "R_Hem", "Center", None, z_offset=z_offset
            )
            cs_pos = [s_pos[0], c_pos[1], c_pos[2]]

            quat = self.skills.calculate_base_to_target_horizontal_quat(
                self.env.right_base, cs_pos
            )

            return s_pos, np.concatenate([cs_pos, quat]), s_idx, c_idx

        # def shoulder_targets(obs, specific_indices=None, z_offset=None):
        #     return self._make_keypoint_grasp_targets(
        #         obs, 'R_Shoulder', 'L_Neck', specific_indices, z_offset=z_offset
        #     )

        success = yield from self.skills.robust_bimanual_grasp(
            close_gripper_value=self.grasp_val,  # T64: 读 yaml grasp_close_val(生产 kinematic yaml 显式 0.015,逐位不变)
            get_targets_func=shoulder_targets,
            max_retries=3,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            retry_inward_step=self.GRASP_XY_OFFSET_STEP,
            grasp_settle_time=self.grasp_settle_time,
            test_lift_height=self.test_lift_height,
            y_retract_offset=0.01,
        )
        if not success:
            return False

        # Phase B: 先抬起来, 让布料自然下垂收缩
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)

        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        l_lift_pos = np.array([l_ee[0], l_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])
        r_lift_pos = np.array([r_ee[0], r_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])

        plan_lift = self.skills.move_dual_arm_to(
            left_target_pose=l_lift_pos,
            right_target_pose=r_lift_pos,
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
            velocity_scale=0.5,
        )
        yield ExecutionStep(
            plan=plan_lift, settle_time=0.3, description="forward_fold_lift"
        )

        # Phase C: 重新观测——布料被抬起后顶点位置已改变
        obs = self.env.get_observation(include_visuals=False)
        self.skills.sync_state(obs)
        cloth_pos = obs.pcd_cloth_truth.copy()

        # 用新观测重新获取关键点位置 (左臂抓的是right_shoulder, 右臂抓的是left_shoulder)
        l_arm_hem = self._get_keypoint_pos(cloth_pos, "right_shoulder")  # 左臂侧下摆
        r_arm_center = self._get_keypoint_pos(cloth_pos, "center")  # 右臂侧中心

        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()
        # Phase D: 基于新观测计算弧线前折

        l_arm_hem[1] = l_ee[1] - 0.02
        r_arm_center[1] = r_ee[1]

        r_arm_center[0] = l_arm_hem[0]

        # Waypoint 2: 折叠 destinations (基于新观测的 hem 位置 + 可调Z偏移)
        fold_dest_z = self.TABLE_HEIGHT * 2
        fold_dest_z = 0.04
        l_fold_dest_pos = np.array([l_arm_hem[0], l_arm_hem[1], fold_dest_z])
        r_fold_dest_pos = np.array([r_arm_center[0], r_arm_center[1], fold_dest_z])
        quat_l_dest = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.left_base, l_fold_dest_pos
        )
        quat_r_dest = self.skills.calculate_base_to_target_horizontal_quat(
            self.env.right_base, r_fold_dest_pos
        )
        l_fold_dest = np.concatenate([l_fold_dest_pos, quat_l_dest])
        r_fold_dest = np.concatenate([r_fold_dest_pos, quat_r_dest])

        logger.info(f"[Forward Fold] L: fold_dest={l_fold_dest}")
        logger.info(f"[Forward Fold] R: fold_dest={r_fold_dest}")

        # Waypoint规划: 从抬起位置 → 弧顶 → 折叠目的地
        plan_arc = self.skills.move_dual_arm_to(
            left_target_pose=[l_fold_dest],
            right_target_pose=[r_fold_dest],
            goal_type="waypoints",
            pose_constraint=[1, 1, 1, 0, 0, 0],
            ee_link="tcp_link_bottom",
        )
        yield ExecutionStep(
            plan=plan_arc, settle_time=0.1, description="forward_fold_arc"
        )

        # 松开
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(
                self.OPEN_SMALL_GRIPPER, self.OPEN_SMALL_GRIPPER, duration=0.3
            ),
            settle_time=0.1,
            description="forward_release",
        )

        # 回撤到上方
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=[l_fold_dest[0], l_fold_dest[1], self.safe_height],
            right_target_pose=[r_fold_dest[0], r_fold_dest[1], self.safe_height],
            pose_constraint=[1, 1, 1, 0, 0, 0],
        )
        yield ExecutionStep(
            plan=plan_retract, settle_time=0.3, description="forward_retract"
        )

        return True

    def _stage_single_vertical_fold(
        self, arm: ArmSide = ArmSide.LEFT
    ) -> Generator[Any, None, bool]:
        """
        单臂竖向折叠: 抓取左右下摆 (L_Hem / R_Hem) 的中心点，折叠到领口 (L_Neck / R_Neck) 的中心点。
        arm: 执行折叠的手臂，默认左臂。
        """
        arm_side_str = "left" if arm == ArmSide.LEFT else "right"
        arm_base = self.env.left_base if arm == ArmSide.LEFT else self.env.right_base

        def hem_target_func(obs, specific_index=None, z_offset=None):
            l_pose_7d, _ = self._make_single_keypoint_grasp_targets(
                obs, "right_hem", arm_side_str, specific_index, z_offset=z_offset
            )
            r_pose_7d, _ = self._make_single_keypoint_grasp_targets(
                obs, "left_hem", arm_side_str, specific_index, z_offset=z_offset
            )
            # 左右下摆抓取位置的中心点，避免布料扭转
            pos = np.mean([l_pose_7d[:3], r_pose_7d[:3]], axis=0)
            pos[2] = self.TABLE_HEIGHT
            quat = l_pose_7d[3:]
            return np.concatenate([pos, quat]), None

        success = yield from self.skills.robust_single_grasp(
            get_target_func=hem_target_func,
            arm=arm,
            other_arm_mode="hold",
            max_retries=3,
            open_gripper_value=self.OPEN_GRIPPER,
            close_gripper_value=self.CLOSE_GRIPPER,
            grasp_depth=self.GRASP_Z_OFFSET,
            retry_depth_step=self.GRASP_Z_OFFSET_STEP,
            grasp_settle_time=self.grasp_settle_time,
            test_lift_height=self.test_lift_height,
            ee_link="tcp_link_bottom",
        )
        if not success:
            return False

        # Phase B: 抬起，让布料自然垂落收缩
        obs = self.reset_interface_state()
        l_ee = obs.eef_pose_left[:3].copy()
        r_ee = obs.eef_pose_right[:3].copy()

        active_ee = l_ee if arm == ArmSide.LEFT else r_ee

        lift_pos = np.array([active_ee[0], active_ee[1], self.FORWARD_FOLD_LIFT_HEIGHT])
        quat_lift = self.skills.calculate_base_to_target_horizontal_quat(
            arm_base, lift_pos
        )
        lift_wp = np.concatenate([lift_pos, quat_lift])

        plan_lift = self.skills.move_to(
            target_pose=lift_wp,
            arm=arm,
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0]],
            ee_link="tcp_link_bottom",
            velocity_scale=0.7,
        )

        yield ExecutionStep(
            plan=plan_lift, settle_time=0.3, description="vertical_fold_lift"
        )

        # Phase C: 重新观测，计算折叠目标——领口左右关键点 (L_Neck / R_Neck) 的中心
        obs = self.reset_interface_state()
        cloth_pos = obs.pcd_cloth_truth.copy()

        l_neck_pos = self._get_keypoint_pos(cloth_pos, "left_collar")
        center_pose = self._get_keypoint_pos(cloth_pos, "center")
        center_pose[2] = self.TABLE_HEIGHT
        center_pose[0] = l_neck_pos[0]  # 固定 x 坐标
        logger.info(f"[Vertical Fold/{arm_side_str}] center_pose={center_pose}")

        quat_fold = self.skills.calculate_base_to_target_horizontal_quat(
            arm_base, center_pose
        )
        fold_wp = np.concatenate([center_pose, quat_fold])

        plan_fold = self.skills.move_to(
            target_pose=fold_wp,
            arm=arm,
            goal_type="waypoints",
            pose_constraint=[[1, 1, 1, 0, 0, 0]],
            ee_link="tcp_link_bottom",
            velocity_scale=0.7,
        )

        yield ExecutionStep(
            plan=plan_fold, settle_time=0.1, description="vertical_fold_arc"
        )

        # 松开
        yield ExecutionStep(
            plan=self.skills.gripper_bimanual_action(
                self.OPEN_SMALL_GRIPPER, self.OPEN_SMALL_GRIPPER, duration=0.3
            ),
            settle_time=0.3,
            description="vertical_fold_release",
        )

        # 回撤
        plan_retract = self.skills.move_dual_arm_to(
            left_target_pose=[0.2, 0.3, self.safe_height],
            right_target_pose=[0.2, -0.3, self.safe_height],
            pose_constraint=[1, 1, 0.8, 0, 0, 0],
        )
        yield ExecutionStep(
            plan=plan_retract, settle_time=0.3, description="vertical_fold_retract"
        )

        return True

    # ========================================
    # Success Check
    # ========================================
    def check_success(self) -> bool:
        cloth_pos = self.env.get_cloth_position()

        l_shoulder = self._get_keypoint_pos(cloth_pos, "left_shoulder")
        r_shoulder = self._get_keypoint_pos(cloth_pos, "right_shoulder")
        l_hem = self._get_keypoint_pos(cloth_pos, "left_hem")
        r_hem = self._get_keypoint_pos(cloth_pos, "right_hem")

        shoulder_center = (l_shoulder + r_shoulder) / 2
        hem_center = (l_hem + r_hem) / 2
        dist = np.linalg.norm(shoulder_center - hem_center)

        threshold = 0.15  # meters
        success = dist < threshold

        logger.info(
            f"[check_success] dist={dist:.4f}, threshold={threshold}, success={success}"
        )

        return success
