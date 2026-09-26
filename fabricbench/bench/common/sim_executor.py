import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from scipy.interpolate import interp1d

from .types import DualPlanResult, ExecutionStep, PlannerErrorCode

if TYPE_CHECKING:  # DataCollector lands with the LeRobot data layer (P4); optional here.
    from .data_collector import DataCollector


class SimExecutor:
    """
    Sim Executor: Handles the time interpolation and physics stepping.
    Acts as the 'Driver' for the environment and triggers the 'Collector'.
    """
    def __init__(
        self,
        env,
        collector: "Optional[DataCollector]" = None,
        physics_freq=500,
        control_freq=25,
        enable_collision_trick: bool = True,
        collision_trick_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            env: The deformable environment (``DeformableEnv``).
            collector: Optional DataCollector instance.
        """
        self.env = env
        self.collector = collector
        self.dt = 1.0 / physics_freq
        # How many physics steps to take before recording one control frame
        self.sim_steps_per_control = int(physics_freq / control_freq)
        logger.info(f"[Collector] Physics: {physics_freq}Hz | Control: {control_freq}Hz "
            f"(Record every {self.sim_steps_per_control} steps)")
        self.last_fail_info = ""

        # Last gripper target we actually COMMANDED (not the measured opening).
        # In PD mode ``env.left_gripper_val`` reads the realized joint angle, so
        # falling back to it re-commands wherever the finger currently sits. On a
        # rigid object that is harmless (the finger is blocked and act==tgt), but
        # on cloth the finger stalls open by the fabric thickness, and each
        # re-command hands that gap back: grip force = kp*(act-tgt) bleeds to zero
        # over the lift and the cloth is dropped. Hold the commanded value instead.
        self._g_cmd_l: float | None = None
        self._g_cmd_r: float | None = None

        # 夹爪碰撞控制器 (防穿模, 默认自动挂载；不支持的 env 安静跳过)
        self.collision_ctrl = None
        self.collision_trick_config = collision_trick_config
        self._collision_trick_supported = hasattr(env, "get_gripper_collision_controller")
        if enable_collision_trick:
            self.enable_collision_trick()

    def enable_collision_trick(self):
        """Enable the base gripper collision trick if the environment supports it."""
        if not self._collision_trick_supported:
            logger.info("[SimExecutor] Collision trick not supported by this env")
            return False

        try:
            self.collision_ctrl = self.env.get_gripper_collision_controller(
                self.collision_trick_config
            )
            if self.collision_ctrl is not None:
                logger.info("[SimExecutor] Base gripper collision trick enabled")
                return True
        except Exception as exc:
            logger.warning(
                f"[SimExecutor] Failed to enable gripper collision trick: {exc}"
            )
        return False

    def _held_gripper(self) -> tuple[float, float]:
        """The gripper target to hold when a plan step carries no gripper column.

        Prefers the last commanded value; only reads the env on the very first
        step, before anything has been commanded. See ``_g_cmd_l``.
        """
        g_l = self._g_cmd_l if self._g_cmd_l is not None else float(self.env.left_gripper_val)
        g_r = self._g_cmd_r if self._g_cmd_r is not None else float(self.env.right_gripper_val)
        return float(g_l), float(g_r)

    def disable_collision_trick(self):
        """Disable the gripper collision trick for this executor."""
        self.collision_ctrl = None
        logger.info("[SimExecutor] Gripper collision trick disabled")

    def set_collision_trick_enabled(self, enabled: bool):
        """Convenience wrapper to toggle the gripper collision trick."""
        if enabled:
            return self.enable_collision_trick()
        self.disable_collision_trick()
        return True


    def execute_plan(self, plan_input: Union[List, ExecutionStep, DualPlanResult, Dict]) -> bool:
        """
        Execute a motion plan (Trajectory) in the simulator.

        Args:
            plan_input: output from PlannerManager with Dual robot arm.
                list -> recursive execution
                ExecutionStep -> support excution + settle time
                DualPlanResult -> pure trajectory (execution only)
                Dict -> Legacy Compatibility
        """
        if isinstance(plan_input, list):
            for step in plan_input:
                if not self.execute_plan(step): return False
            return True

        self.last_fail_info = ""

       # Data Adapter
        plan_res = None
        settle_time = 0.0
        # Case A:  ExecutionStep
        if hasattr(plan_input, 'plan') and hasattr(plan_input, 'settle_time'):
            plan_res = plan_input.plan
            settle_time = plan_input.settle_time

        # Case B: 纯轨迹 DualPlanResult
        elif hasattr(plan_input, 'left') and hasattr(plan_input, 'status'):
            plan_res = plan_input
            settle_time = 0.0

        # Case C: 旧字典兼容
        elif isinstance(plan_input, dict):
            if plan_input.get('status') != 'Success': return False
            plan_res = plan_input # 字典本身充当 plan 对象
            settle_time = plan_input.get('settle_time', 0.0)

        else:
            logger.error(f"Unknown plan type: {type(plan_input)}")
            return False

        # ==========================================================
        # 3. check success
        # ==========================================================
        is_success = False
        if hasattr(plan_res, 'success'): is_success = plan_res.success
        elif isinstance(plan_res, dict): is_success = (plan_res.get('status') == 'Success')

        if not is_success:
            error_code = getattr(plan_res, 'error_code', PlannerErrorCode.UNKNOWN)
            error_msg = getattr(plan_res, 'error_msg', '')

            code_str = error_code.name if hasattr(error_code, 'name') else str(error_code)
            self.last_fail_info = f"{code_str}"

            logger.error(
                f"[PLAN_FAILED] ErrorCode={error_code.name} | Detail={error_msg}"
            )
            return False

        # ==========================================================
        # 4. Physics Execution
        # ==========================================================

        # --- 2. Interpolation Setup ---
        # Extract Left/Right trajectories
        if isinstance(plan_res, dict):
            left_traj = plan_res['left']
            right_traj = plan_res['right']
        else:
            left_traj = plan_res.left
            right_traj = plan_res.right
        duration = max(left_traj.time[-1], right_traj.time[-1])

        # Seed the final commanded targets up front so zero-duration / single-sample
        # plans can still settle correctly without entering the interpolation loop.
        q_l_cmd = np.array(self.env.left_qpos, copy=True)
        q_r_cmd = np.array(self.env.right_qpos, copy=True)
        g_l_cmd, g_r_cmd = self._held_gripper()

        if len(left_traj.position) > 0:
            q_l_full_final = np.asarray(left_traj.position[-1])
            n_l = len(self.env.idx_l_arm)
            q_l_cmd = q_l_full_final[:n_l]
            if len(q_l_full_final) > n_l:
                g_l_cmd = float(q_l_full_final[n_l])

        if len(right_traj.position) > 0:
            q_r_full_final = np.asarray(right_traj.position[-1])
            n_r = len(self.env.idx_r_arm)
            q_r_cmd = q_r_full_final[:n_r]
            if len(q_r_full_final) > n_r:
                g_r_cmd = float(q_r_full_final[n_r])

        q_l_final = np.array(q_l_cmd, copy=True)
        q_r_final = np.array(q_r_cmd, copy=True)
        g_l_final = float(g_l_cmd)
        g_r_final = float(g_r_cmd)

        # A hold/gripper/no-op plan may legitimately contain only a single sample at
        # t=0. In that case there is nothing to interpolate; just execute settle/hold
        # with the final target parsed above.
        if duration <= 0 or len(left_traj.time) < 2 or len(right_traj.time) < 2:
            if settle_time > 0:
                self.hold(
                    settle_time,
                    q_l_cmd=q_l_cmd, q_r_cmd=q_r_cmd,
                    g_l_cmd=g_l_cmd, g_r_cmd=g_r_cmd,
                )
            return True

        # Create linear interpolators (Time -> Joint Positions)
        # fill_value="extrapolate" ensures we don't crash if sim_time slightly exceeds plan time
        f_l = interp1d(left_traj.time, left_traj.position, axis=0, kind='linear', fill_value="extrapolate")
        f_r = interp1d(right_traj.time, right_traj.position, axis=0, kind='linear', fill_value="extrapolate")

        # --- 3. Physics Loop (Trajectory) ---
        sim_time = 0.0
        step_counter = 0

        # Run Trajectory
        while sim_time < duration:
            # A. Calculate Action (Target) at current time
            t_query = sim_time
            q_l_full = f_l(t_query)
            q_r_full = f_r(t_query)

            # B. Parse Full Vector (Joints + Gripper)
            # Assumption: Last element is gripper, previous N are arm joints
            n_l = len(self.env.idx_l_arm)
            q_l_cmd = q_l_full[:n_l]
            g_l_cmd = q_l_full[n_l] if len(q_l_full) > n_l else 0.0

            n_r = len(self.env.idx_r_arm)
            q_r_cmd = q_r_full[:n_r]
            g_r_cmd = q_r_full[n_r] if len(q_r_full) > n_r else 0.0

            # C. Step Physics
            self._g_cmd_l, self._g_cmd_r = float(g_l_cmd), float(g_r_cmd)
            self.env.step(q_l_cmd, q_r_cmd, g_l_cmd, g_r_cmd)
            if self.collision_ctrl:
                self.collision_ctrl.update(
                    float(self.env.left_gripper_val),
                    float(self.env.right_gripper_val),
                )

            # C2. 实时打印夹爪+碰撞状态 (每100步≈0.2s)
            if self.collision_ctrl and step_counter % 100 == 0:
                col_l = "ON" if self.collision_ctrl.is_collision_enabled("left") else "OFF"
                col_r = "ON" if self.collision_ctrl.is_collision_enabled("right") else "OFF"
                sys.stdout.write(
                    f"\r  Grip L:{g_l_cmd:.3f} R:{g_r_cmd:.3f} | "
                    f"Col L:{col_l} R:{col_r}   "
                )
                sys.stdout.flush()

            # D. Record (If collector is present)
            # We construct the action dict here. The collector handles frequency downsampling.
            self._try_record({
                'q_l': q_l_cmd, 'q_r': q_r_cmd,
                'g_l': g_l_cmd, 'g_r': g_r_cmd
            })

            # E. Render step callback (for trajectory recording)
            if hasattr(self, '_render_step_callback') and self._render_step_callback:
                self._render_step_callback()

            sim_time += self.dt
            step_counter += 1

        # ---  Settle / Hold (Optional) ---
        # Pass the trajectory's final commanded targets so PD-mode controllers
        # keep producing torque toward those targets during settle. (In
        # kinematic mode the qpos is overridden anyway, so passing them is a
        # no-op for backward compatibility.)
        if settle_time > 0:
            self.hold(
                settle_time,
                q_l_cmd=q_l_final, q_r_cmd=q_r_final,
                g_l_cmd=g_l_final, g_r_cmd=g_r_final,
            )

        return True

    def hold(self, duration: float,
             q_l_cmd: Optional[np.ndarray] = None,
             q_r_cmd: Optional[np.ndarray] = None,
             g_l_cmd: Optional[float] = None,
             g_r_cmd: Optional[float] = None):
        """Hold robot for ``duration`` seconds.

        Each step receives the same command targets as PD/kinematic input.
        The defaults preserve the original behaviour of holding at the
        current actual qpos. Callers driving PD control should pass the
        trajectory's last commanded target so PD keeps producing force
        (e.g. a gripper-close command keeps the fingers pressing the bar
        even after the trajectory has finished).
        """
        if duration <= 0: return

        if q_l_cmd is None: q_l_cmd = self.env.left_qpos
        if q_r_cmd is None: q_r_cmd = self.env.right_qpos
        if g_l_cmd is None or g_r_cmd is None:
            held_l, held_r = self._held_gripper()
            if g_l_cmd is None: g_l_cmd = held_l
            if g_r_cmd is None: g_r_cmd = held_r

        sim_time = 0.0
        hold_step = 0
        self._g_cmd_l, self._g_cmd_r = float(g_l_cmd), float(g_r_cmd)

        while sim_time < duration:
            # 2. Step Physics (Keep targets constant)
            self.env.step(q_l_cmd, q_r_cmd, g_l_cmd, g_r_cmd)
            if self.collision_ctrl:
                self.collision_ctrl.update(
                    float(self.env.left_gripper_val),
                    float(self.env.right_gripper_val),
                )

            if self.collision_ctrl and hold_step % 100 == 0:
                col_l = "ON" if self.collision_ctrl.is_collision_enabled("left") else "OFF"
                col_r = "ON" if self.collision_ctrl.is_collision_enabled("right") else "OFF"
                sys.stdout.write(
                    f"\r  [Hold] Grip L:{g_l_cmd:.3f} R:{g_r_cmd:.3f} | "
                    f"Col L:{col_l} R:{col_r}   "
                )
                sys.stdout.flush()

            # 3. Record
            self._try_record({
                'q_l': q_l_cmd, 'q_r': q_r_cmd,
                'g_l': g_l_cmd, 'g_r': g_r_cmd
            })

            # 4. Render step callback
            if hasattr(self, '_render_step_callback') and self._render_step_callback:
                self._render_step_callback()

            sim_time += self.dt
            hold_step += 1

    def _try_record(self, action_dict):
        """Helper to trigger recording safely."""
        if self.collector:
            # The collector checks 'current_stage_name' internally.
            # If it's None (recording stopped), it simply returns.
            self.collector.record_frame(self.env, action_dict)
