"""Passively collect firmware-linked dual-PiPER demonstrations.

The left master/follower pair shares ``can_left`` and the right pair shares
``can_right``.  This entry point reads follower feedback and cameras only; it
never sends robot commands.

Example on the robot machine::

    python -m deformablebench.bench.collect_real \
      --robot.type=dual_piper \
      --robot.left_port=can_left --robot.right_port=can_right \
      --robot.cameras="{static_cam: {...}, left_hand_cam: {...}, right_hand_cam: {...}}" \
      --task=grasp_silk_bimanual --repo_id=local/grasp_silk_real_v1
"""

import logging
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from deformablebench._env import ensure_clean_ld_path

ensure_clean_ld_path()

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot.cameras.realsense.configuration_realsense import (  # noqa: F401, E402
    RealSenseCameraConfig,
)
from lerobot.configs import parser  # noqa: E402
from lerobot.robots import RobotConfig, make_robot_from_config  # noqa: E402
from lerobot.robots import dual_piper  # noqa: F401, E402

from deformablebench.bench.data.collect_real import (  # noqa: E402
    PassiveFollowerEpisode,
    open_real_dataset,
)
from deformablebench.bench.real.camera import (  # noqa: E402
    DEFAULT_HUE,
    DEFAULT_POWER_LINE_FREQUENCY,
    DEFAULT_WHITE_BALANCE,
    CameraColorConfig,
    apply_camera_color_settings,
)
from deformablebench.paths import REPO_ROOT  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class CollectRealConfig:
    robot: RobotConfig | None = None
    task: str | None = None
    task_instruction: str | None = None
    repo_id: str | None = None
    lerobot_root: str = "data/lerobot_dataset"
    num_episodes: int = 1
    fps: int = 25
    episode_time_s: float = 120.0
    resume: bool = False
    # ---- 相机 RGB 参数（曝光/增益/白平衡/工频/色相）----------------------------
    # 实现与默认值在 deformablebench.bench.real.camera，采集与推理共用一份
    # （分成两份实现正是 2026-09 那次色彩域漂移的根因）。
    #
    # **`manual_exposure` 是总开关**：不传（None）=> 一个相机参数都不动，
    # 相机保持自动曝光 + 自动白平衡，下面 auto_wb/white_balance/... 全部不生效。
    # 也就是说 `python -m deformablebench.bench.collect_real` 裸跑 = 旧行为。
    #
    # 但**实际采集入口是 `tools/collect_real_teleop.sh`，它默认会传**
    #     --manual_exposure=15 --manual_gain=64 --auto_wb=false --white_balance=6400
    # 所以走脚本采集时，默认是**固定白平衡**；脚本加 `--auto-awb` 回到自动。
    #
    # 为什么默认固定：AWB 在 LED 下持续 hunt，帧间 std≈5.2，而固定 WB≈0.15。
    # AWB 那份抖动是**时变**的，训练侧的 ColorJitter 覆盖不了（jitter 治的是固定
    # 偏移）；逐台相机的偏色才是硬件级固定偏移，交给 jitter。15/64/6400 来自
    # deformable_bench Round 5 标定（2026-04-02），见 real/camera.py 文件头。
    #
    # ⚠ **色彩域配对规则**：固定 WB 与 AWB 是两个色彩域，**不能混进同一个数据集**，
    # 评测时也必须与该 checkpoint 训练数据所用的那一档一致：
    #     grasp_silk_real_v1（20 集，2026-09-06 之前）  -> AWB，评测用 --auto_wb=true
    #     此后新采的                                    -> 固定 WB，评测用默认
    manual_exposure: int | None = None   # None = 自动曝光，且下面几项一概不生效
    manual_gain: int | None = None
    auto_wb: bool = False                # False = 固定白平衡（仅在 manual_exposure 已设时）
    white_balance: int = DEFAULT_WHITE_BALANCE          # 6400 K
    power_line_frequency: int = DEFAULT_POWER_LINE_FREQUENCY   # 1 = 50Hz
    hue: int = DEFAULT_HUE                              # 0
    start_immediately: bool = False
    # Reuse LeRobot's Rerun UI to preview cameras and follower state while recording.
    display_data: bool = False


def _task_instruction(task: str | None, override: str | None) -> str:
    if override:
        return override
    if not task:
        raise ValueError("--task is required unless --task_instruction is provided")

    import yaml

    path = REPO_ROOT / "src" / "deformablebench" / "bench" / "config" / "task" / f"{task}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"task config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    instruction = raw.get("task_description")
    if not instruction:
        raise ValueError(f"{path} has no task_description")
    return str(instruction)


def _apply_camera_settings(robot, cfg: "CollectRealConfig") -> None:
    """采集侧与 real_eval 共用同一份实现，见 bench/real/camera.py。"""
    apply_camera_color_settings(
        robot,
        CameraColorConfig(
            exposure=cfg.manual_exposure,
            gain=cfg.manual_gain,
            auto_wb=cfg.auto_wb,
            white_balance=cfg.white_balance,
            power_line_frequency=cfg.power_line_frequency,
            hue=cfg.hue,
        ),
    )


class CaptureKeys:
    """Non-blocking single-key controls that work in an SSH terminal."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._old_settings = None

    def __enter__(self):
        if self.stream.isatty():
            self._old_settings = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno())
        return self

    def poll(self) -> str | None:
        if self._old_settings is None:
            return None
        ready, _, _ = select.select([self.stream], [], [], 0.0)
        return self.stream.read(1).lower() if ready else None

    def __exit__(self, exc_type, exc, traceback):
        if self._old_settings is not None:
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._old_settings)


def _start_rerun_preview() -> Callable[[dict], None]:
    """Start the same live Rerun viewer used by ``lerobot-record``."""
    try:
        from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    except ModuleNotFoundError as exc:
        if exc.name == "rerun":
            raise RuntimeError(
                "--display_data=true requires rerun-sdk; install the version pinned by "
                "the project in the real-robot environment"
            ) from exc
        raise

    init_rerun(session_name="real_teleop_recording")
    logger.info("Rerun live preview enabled")

    def log_observation(observation: dict) -> None:
        # Visualization is a read-only side channel. Dataset action alignment remains
        # action[t] = follower_state[t + 1] inside PassiveFollowerEpisode.
        log_rerun_data(observation=observation)

    return log_observation


def capture_episode(
    *,
    robot,
    dataset,
    camera_names: list[str],
    task: str,
    fps: int,
    episode_time_s: float,
    key_source_factory: Callable[[], CaptureKeys] = CaptureKeys,
    observation_callback: Callable[[dict], None] | None = None,
) -> tuple[PassiveFollowerEpisode, str, float]:
    """Capture until timeout or a terminal key; return episode and requested disposition."""
    episode = PassiveFollowerEpisode(dataset, camera_names, task)
    period_s = 1.0 / float(fps)
    start_t = time.perf_counter()
    disposition = "review"

    print("Recording: SPACE=finish, r=discard/re-record, q=discard/quit", flush=True)
    with key_source_factory() as keys:
        while time.perf_counter() - start_t < episode_time_s:
            loop_t = time.perf_counter()
            observation = robot.get_observation()
            episode.add_observation(observation)
            if observation_callback is not None:
                observation_callback(observation)

            key = keys.poll()
            if key == " ":
                break
            if key == "r":
                disposition = "rerecord"
                break
            if key == "q":
                disposition = "quit"
                break

            remaining = period_s - (time.perf_counter() - loop_t)
            if remaining > 0:
                time.sleep(remaining)

    elapsed_s = time.perf_counter() - start_t
    episode.seal()
    return episode, disposition, elapsed_s


def _review_episode(episode: PassiveFollowerEpisode) -> str:
    if not sys.stdin.isatty():
        return "save"
    while True:
        answer = input(
            f"Captured {episode.num_frames} frames. ENTER=save, r=discard/re-record, q=discard/quit: "
        ).strip().lower()
        if answer in {"", "s"}:
            return "save"
        if answer == "r":
            return "rerecord"
        if answer == "q":
            return "quit"


def collect(cfg: CollectRealConfig) -> int:
    if cfg.robot is None or cfg.robot.type != "dual_piper":
        raise ValueError("--robot.type=dual_piper is required")
    if not cfg.repo_id:
        raise ValueError("--repo_id is required; use a new id for every formal collection")
    if cfg.num_episodes <= 0 or cfg.fps <= 0 or cfg.episode_time_s <= 0:
        raise ValueError("num_episodes, fps, and episode_time_s must all be positive")
    if getattr(cfg.robot, "point_cloud", None) and cfg.robot.point_cloud.enabled:
        raise ValueError("collect_real currently records follower state + RGB only; disable point_cloud")

    instruction = _task_instruction(cfg.task, cfg.task_instruction)
    dataset_root = Path(cfg.lerobot_root) / cfg.repo_id
    robot = make_robot_from_config(cfg.robot)
    dataset = None
    saved = 0
    observation_callback = None

    try:
        robot.connect(calibrate=False)
        _apply_camera_settings(robot, cfg)
        dataset, camera_names = open_real_dataset(
            robot=robot,
            root=dataset_root,
            repo_id=cfg.repo_id,
            fps=cfg.fps,
            resume=cfg.resume,
        )
        if cfg.display_data:
            observation_callback = _start_rerun_preview()

        logger.info("dataset=%s", dataset_root)
        logger.info("task=%r", instruction)
        logger.info("CAN buses=%s,%s", cfg.robot.left_port, cfg.robot.right_port)

        while saved < cfg.num_episodes:
            episode_number = int(dataset.meta.total_episodes)
            if not cfg.start_immediately and sys.stdin.isatty():
                answer = input(
                    f"Reset the scene for episode {episode_number}, then press ENTER to record (q=quit): "
                ).strip().lower()
                if answer == "q":
                    break

            episode, disposition, elapsed_s = capture_episode(
                robot=robot,
                dataset=dataset,
                camera_names=camera_names,
                task=instruction,
                fps=cfg.fps,
                episode_time_s=cfg.episode_time_s,
                observation_callback=observation_callback,
            )
            if disposition == "review":
                disposition = _review_episode(episode)

            if disposition == "save":
                frame_count = episode.commit()
                saved += 1
                actual_fps = frame_count / max(elapsed_s, 1e-9)
                logger.info(
                    "saved episode %d: frames=%d elapsed=%.2fs actual_fps=%.2f",
                    episode_number,
                    frame_count,
                    elapsed_s,
                    actual_fps,
                )
            else:
                episode.discard()
                logger.info("discarded episode %d", episode_number)
                if disposition == "quit":
                    break
    except KeyboardInterrupt:
        logger.info("collection interrupted")
        if dataset is not None:
            dataset.clear_episode_buffer()
    except Exception:
        if dataset is not None:
            dataset.clear_episode_buffer()
        raise
    finally:
        if dataset is not None and hasattr(dataset, "finalize"):
            dataset.finalize()
        # A camera failure makes ``is_connected`` false even while the CAN
        # interfaces are still open, so key cleanup off the robot connection.
        if getattr(robot, "is_robot_connected", robot.is_connected):
            robot.disconnect()

    return saved


@parser.wrap()
def main(cfg: CollectRealConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    saved = collect(cfg)
    logger.info("collection complete: +%d episode(s)", saved)


if __name__ == "__main__":
    main()
