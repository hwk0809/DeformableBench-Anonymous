"""RealSense RGB 参数的唯一入口，采集与推理共用。

分成两份实现是 2026-09 那次色彩域漂移的根因：`collect_real.py` 从
`real_eval.py` 抄 `set_manual_exposure` 时漏掉了白平衡那段，于是采集侧的
白平衡一直落在相机出厂默认（AWB=on / WB=4600），而推理侧显式开着 AWB。
两边碰巧都是"AWB 开"才没出事。任何一侧要改相机参数，改这里。

默认值来自 `deformable_bench/docs/sim_real_gap_analysis.md` 的
Selected Camera Settings（Round 5，2026-04-02 标定）:

    exposure=15, gain=64, AWB=off, WB=6400, PLF=50Hz

那次标定的取舍是"在不闪烁的前提下把 WB 推到最高"——WB 越高越不偏蓝，
但 6200 起手腕相机开始不稳、6500 全相机固件抽风，6400 是稳定区上限。
2026-09-06 用"距离仿真集色彩"这把尺子复测过一轮，结论不变：

- 固定 WB 帧间 std ≈ 0.15，AWB ≈ 5.2（AWB 在 LED 下持续 hunt）；
- 逐台相机的偏色是硬件级的（static 偏蓝、两个手腕偏红，R-G 极差约 40），
  跨姿态、跨月份稳定 → 是**固定**偏移，模型学得掉，交给训练侧的
  ColorJitter 覆盖，不在采集时用 per-camera hue 硬补（那会把三个魔法
  数字绑死在当前光照上）。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 标定选定值，改这里等于同时改采集和推理。
DEFAULT_WHITE_BALANCE = 6400
DEFAULT_POWER_LINE_FREQUENCY = 1  # 0=disabled, 1=50Hz, 2=60Hz, 3=auto
DEFAULT_HUE = 0


@dataclass
class CameraColorConfig:
    """三路 RGB 相机的颜色相关参数（曝光/增益/白平衡/工频/色相）。"""

    exposure: int | None = None
    gain: int | None = None
    # False（默认）= 固定白平衡，可复现;True = 自动白平衡，复现 2026-09-06
    # 之前的历史采集与评测（那时两侧都跑在 AWB 上）。
    auto_wb: bool = False
    white_balance: int = DEFAULT_WHITE_BALANCE
    power_line_frequency: int = DEFAULT_POWER_LINE_FREQUENCY
    hue: int = DEFAULT_HUE


def apply_camera_color_settings(robot, cfg: CameraColorConfig) -> None:
    """把 ``cfg`` 应用到 robot 的每一路 RealSense，并回读实际生效值。

    ``exposure is None`` 表示整体走自动曝光,此时白平衡也不碰——保持
    "没显式要求就一个都不动"的语义,与旧行为一致。

    回读是有意为之:RealSense 的这些 option 是**设备上持久的**,跨进程保留。
    只写不读的话,某个程序留下的残留值会静默地进入下一次采集,而日志里看不出来
    ——2026-09 那次就是事后无法确定"录制当时白平衡到底是多少"。
    """
    if cfg.exposure is None:
        logger.info("manual_exposure 未设置：相机保持自动曝光/自动白平衡，未做任何改动")
        return

    try:
        import pyrealsense2 as rs
    except ImportError:
        logger.warning("pyrealsense2 不可用；相机参数未应用")
        return

    for camera_name, camera in robot.cameras.items():
        try:
            device = camera.rs_profile.get_device()
        except AttributeError:
            # 非 RealSense 相机（OpenCV 等）没有 rs_profile，跳过。
            continue

        try:
            for sensor in device.query_sensors():
                if sensor.get_info(rs.camera_info.name) != "RGB Camera":
                    continue

                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, cfg.exposure)
                if cfg.gain is not None:
                    sensor.set_option(rs.option.gain, cfg.gain)
                sensor.set_option(rs.option.power_line_frequency, cfg.power_line_frequency)
                sensor.set_option(rs.option.hue, cfg.hue)

                if cfg.auto_wb:
                    sensor.set_option(rs.option.enable_auto_white_balance, 1)
                else:
                    sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    sensor.set_option(rs.option.white_balance, cfg.white_balance)

                _log_readback(rs, sensor, camera_name)
                break
            else:
                logger.warning("[%s] 未找到 RGB Camera sensor；参数未应用", camera_name)
        except Exception as exc:
            logger.warning("[%s] 相机参数应用失败: %s", camera_name, exc)


def _log_readback(rs, sensor, camera_name: str) -> None:
    """回读并打印实际生效的 ISP 状态，让每次跑的日志自带取证信息。"""
    def get(option):
        try:
            return f"{sensor.get_option(option):.0f}"
        except Exception:
            return "n/a"

    awb = get(rs.option.enable_auto_white_balance)
    wb_note = "（AWB 开启时该值不代表实际增益）" if awb == "1" else ""
    logger.info(
        "[%s] 实际生效: exposure=%s gain=%s auto_exposure=%s auto_wb=%s "
        "white_balance=%s%s power_line_freq=%s hue=%s",
        camera_name,
        get(rs.option.exposure),
        get(rs.option.gain),
        get(rs.option.enable_auto_exposure),
        awb,
        get(rs.option.white_balance),
        wb_note,
        get(rs.option.power_line_frequency),
        get(rs.option.hue),
    )
