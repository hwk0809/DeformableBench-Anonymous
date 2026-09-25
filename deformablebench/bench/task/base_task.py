import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Protocol, runtime_checkable
from omegaconf import DictConfig
from loguru import logger
from contextlib import contextmanager

# Import your custom classes for type hinting
from deformablebench.bench.common.success_checker import SuccessChecker
from deformablebench.bench.robot.skills.closed_loop_skills import ClosedLoopSkills
from deformablebench.bench.robot.planning.planner_manager import PlannerManager

# Environment TODO: support more env
@runtime_checkable
class EnvProtocol(Protocol):
    def get_observation(self) -> Any: ...
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Any: ...

# The newton pipeline drives tasks through EnvProtocol; type hints use it directly.
EnvType = EnvProtocol


class BaseTask(ABC):
    """
    Abstract Base Class for all tasks.
    Acts as the 'Director' or 'Brain' of the system.
    Responsible for high-level logic, not low-level execution.
    """

    def __init__(
        self, 
        env: EnvProtocol, 
        planner_manager: PlannerManager,
        collector=None, 
        cfg: Optional[DictConfig] = None
    ) -> None:
        """
        Initialize the Task logic.

        Args:
            env: Instance of the MuJoCo environment.
            planner_manager: Manager for motion planning (Curobo/Toppra).
            cfg: Hydra configuration object containing 'task', 'robot', 'env' specs.
        """
        self.env = env
        self.cfg = cfg if cfg is not None else {}
        self.collector = collector
        self.planner_manager = planner_manager
        
        # 1. Initialize Skill Library (Composition over Inheritance)
        # Pass the robot config subset to skills for gripper mapping logic
        robot_cfg = self.cfg.get("robot", {}) if isinstance(self.cfg, dict) or isinstance(self.cfg, DictConfig) else {}
        self.skills = ClosedLoopSkills(planner_manager, env.get_observation, robot_cfg)
        
        # 2. Build SuccessChecker from yaml eval.success_check (if present)
        eval_cfg = self.cfg.get("eval", {}) if self.cfg else {}
        sc_cfg = eval_cfg.get("success_check", None) if eval_cfg else None
        self._success_checker = SuccessChecker(sc_cfg, env) if sc_cfg else None

        # 3. Task Metadata Storage
        # Stores runtime info like target positions, descriptions, etc.
        self.task_meta: Dict[str, Any] = {}

    def reset_interface_state(self):
        """
        Reset any internal state of the task or skills before starting a new episode.
        This can be called by the main loop before each episode.
        """
        obs = self.env.get_observation(include_visuals=False)  # Get fresh observation
        self.skills.sync_state(obs)

        return obs
    
    @abstractmethod
    def plan_episode(self, episode_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Core Abstract Method: Generate the expert demonstration trajectory.
        
        Args:
            episode_id: The current episode index (useful for seeding or specific variations).
            
        Returns:
            A list of stage dictionaries describing the plan, e.g.:
            [
                {"name": "stage1", "plan": PlanResult, "settle_time": 0.5},
                ...
            ]
            Returns None if planning fails.
        """
        pass

    def check_success(self) -> bool:
        """Determine if the task is successfully completed.

        Default implementation uses yaml ``eval.success_check`` config via
        :class:`SuccessChecker`. Subclasses with stage-flag logic (fold, flatten)
        should override this method.
        """
        if self._success_checker is not None:
            return self._success_checker.check()
        logger.warning("[BaseTask] No success_check config and no override — returning False")
        return False

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata to be saved with the dataset (e.g., text descriptions).
        
        Returns:
            Dictionary of metadata. Defaults to self.task_meta.
        """
        return self.task_meta

    # --- Utility Methods ---
    
    def get_current_obs(self) -> Any:
        """Shortcut to get current environment observation."""
        return self.env.get_observation()
    
    @contextmanager
    def recording(self, stage_name: str):
        logger.info(f"🔴 [REC] Start recording: {stage_name}")
        if self.collector:
            self.collector.start_stage(stage_name)
        
        try:
            yield 
        finally:
            logger.info(f"⬛ [STOP] Stop recording: {stage_name}")
            if self.collector:
                self.collector.stop_stage()