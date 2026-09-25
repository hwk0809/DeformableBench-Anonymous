"""Benchmark orchestrator + reporting for closed-loop sim eval.

Two modes, one file:

  1. RUN a matrix (``--config <yaml>``): sweeps ``model x task x randomization``,
     shelling ``bench.eval_sim`` once per cell. It is the *outer supervisor* — each
     cell runs with an outer ``timeout_per_run`` force-kill (SIGKILL the whole
     eval_sim process group, which owns Kit), GPU-idle hygiene between cells, and
     auto-resume: on timeout/crash it relaunches ``eval_sim --resume`` so completed
     episodes are skipped. eval_sim's own per-episode ``--timeout`` is the inner tier.
     Cell outputs are organized as
     ``output_dir/<task>/<run_id>/<preset>/<model>/``.

  2. REPORT only (``--eval_root`` / ``--eval_dirs``): does NOT run anything — scans
     finished eval runs and emits a comparison table (markdown + csv + json) with
     binomial success rate + Wilson 95% CI.

    # run a matrix from a config, then report
    .venv/bin/python -m deformablebench.bench.benchmark --config configs/benchmark/silk_v1.yaml
    # aggregate already-finished runs
    .venv/bin/python -m deformablebench.bench.benchmark --eval_root data/eval_out

Config schema (yaml): see ``MatrixConfig`` / ``ModelSpec`` / ``TaskEntry`` below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

EVAL_MODULE = "deformablebench.bench.eval_sim"


# ===========================================================================
# Config dataclasses (loaded from yaml)
# ===========================================================================
@dataclass
class ModelSpec:
    """One policy to benchmark (a matrix row)."""

    name: str                       # display name / output subdir, e.g. "ACT-30k"
    policy_type: str = "lerobot"    # backend: lerobot | lerobot_remote | gr00t | lingbot
    checkpoint: str = ""
    n_action_steps: int | None = None
    # Optional per-model overrides. Model overrides win over task/global values
    # because policy backends can have very different normal runtimes.
    episode_timeout: int | None = None
    timeout_per_run: int | None = None
    # Verbatim passthrough flags for the backend, e.g. for lerobot_remote:
    #   {"--server_python": ".../envs/lerobot_v042/bin/python", "--server_port": "5560"}
    extra_args: dict = field(default_factory=dict)


@dataclass
class TaskEntry:
    """One task, optionally swept over several randomization presets (matrix cols)."""

    task: str
    # Difficulty presets (config/randomization/<name>.yaml). ``null`` = task's own DR.
    randomizations: list = field(default_factory=lambda: [None])
    num_episodes: int | None = None    # per-task override of the global default
    sim_duration: float | None = None
    max_steps: int | None = None
    episode_timeout: int | None = None
    timeout_per_run: int | None = None
    asset_pool: str | None = None
    asset_seed: int = 20260915


@dataclass
class MatrixConfig:
    models: list[ModelSpec]
    tasks: list[TaskEntry]
    output_dir: str = "data/benchmark_results"
    run_id: str = "auto"            # "auto" -> YYYYMMDD_HHMMSS
    device: str = "cuda"
    render: str = "kit"
    width: int = 1280
    height: int = 720
    save_video: bool = False
    video_camera: str = "static_cam"  # comma-separated cameras or "all"
    # Eval defaults (per-task overrides win)
    num_episodes: int = 20
    sim_duration: float = 12.0
    check_interval: float = 1.0
    ema_alpha: float = 0.0
    clip_gripper: bool = False
    grasp_close_val: float = 0.0
    eval_seed: int = 41            # ONE held-out seed for all axes; data uses 0/42/43,
                                   # so 42 (the pre-2026-07-26 default) leaked on 4 tasks
    # Two-tier timeout: inner kills one stuck episode; outer kills the whole cell.
    episode_timeout: int = 400          # eval_sim --timeout (inner, per episode)
    timeout_per_run: int = 3600         # outer force-kill per (model x task x preset)
    max_retries: int = 2                # relaunch (with --resume) on timeout/crash
    resume: bool = True
    # GPU hygiene between cells / before a resume relaunch
    min_gpu_free_mib: int = 6000
    gpu_wait_max_s: int = 120


def _load_config(path: str) -> MatrixConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    models = [ModelSpec(**m) for m in raw.pop("models", [])]
    tasks = [TaskEntry(**t) for t in raw.pop("tasks", [])]
    if not models or not tasks:
        raise ValueError(f"{path}: config needs non-empty 'models' and 'tasks'")
    unknown = set(raw) - set(MatrixConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = MatrixConfig(models=models, tasks=tasks, **raw)
    if not cfg.run_id or cfg.run_id == "auto":
        cfg.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg


# ===========================================================================
# GPU hygiene (ported from deformable_bench's benchmark.py)
# ===========================================================================
def _query_gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5)
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def _kill_ghost_kits() -> None:
    """Best-effort: kill leftover Kit render / eval_sim children holding the GPU.

    Scoped to deformablebench's Kit render script + eval_sim module so a *concurrent*
    benchmark's workers are not affected. The main cleanup is the per-cell
    process-group SIGKILL; this is a secondary net on GPU-idle timeout.
    """
    for pat in ("omniverse_kit_render", EVAL_MODULE):
        subprocess.run(["pkill", "-9", "-f", pat], check=False, capture_output=True)


def _wait_gpu_idle(min_free_mib: int, max_wait_s: int, poll_s: float = 2.0) -> bool:
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        free = _query_gpu_free_mib()
        if free is None:      # nvidia-smi unavailable → just proceed
            return True
        if free >= min_free_mib:
            return True
        time.sleep(poll_s)
    print(f"  [gpu-wait] not idle in {max_wait_s}s (free={_query_gpu_free_mib()} MiB, "
          f"need={min_free_mib}) — killing ghost Kit/eval...")
    _kill_ghost_kits()
    time.sleep(5)
    free = _query_gpu_free_mib()
    ok = free is None or free >= min_free_mib
    print(f"  [gpu-wait] after cleanup: {free} MiB free ({'ok' if ok else 'still low, may OOM'})")
    return ok


# ===========================================================================
# Matrix runner
# ===========================================================================
def _preset_name(preset) -> str:
    return str(preset) if preset is not None else "base"


def _task_run_dir(cfg: MatrixConfig, task: TaskEntry) -> Path:
    return Path(cfg.output_dir) / task.task / cfg.run_id


def _report_dir(cfg: MatrixConfig) -> Path:
    return Path(cfg.output_dir) / "_reports" / cfg.run_id


def _cell_dir(cfg: MatrixConfig, model: ModelSpec, task: TaskEntry, preset) -> Path:
    return _task_run_dir(cfg, task) / _preset_name(preset) / model.name


def _cell_episode_timeout(cfg: MatrixConfig, model: ModelSpec, task: TaskEntry) -> int:
    if model.episode_timeout is not None:
        return int(model.episode_timeout)
    if task.episode_timeout is not None:
        return int(task.episode_timeout)
    return int(cfg.episode_timeout)


def _cell_timeout_per_run(cfg: MatrixConfig, model: ModelSpec, task: TaskEntry) -> int:
    if model.timeout_per_run is not None:
        return int(model.timeout_per_run)
    if task.timeout_per_run is not None:
        return int(task.timeout_per_run)
    return int(cfg.timeout_per_run)


def build_cmd(cfg: MatrixConfig, model: ModelSpec, task: TaskEntry, preset, cell: Path) -> list[str]:
    n_ep = task.num_episodes if task.num_episodes is not None else cfg.num_episodes
    sim_dur = task.sim_duration if task.sim_duration is not None else cfg.sim_duration
    cmd = [sys.executable, "-m", EVAL_MODULE,
           "--task", task.task, "--policy_type", model.policy_type,
           "--checkpoint", model.checkpoint, "--render", cfg.render,
           "--output_dir", str(cell), "--device", cfg.device,
           "--width", str(cfg.width), "--height", str(cfg.height),
           "--num_episodes", str(n_ep), "--sim_duration", str(sim_dur),
           "--check_interval", str(cfg.check_interval),
           "--ema_alpha", str(cfg.ema_alpha),
           "--grasp_close_val", str(cfg.grasp_close_val),
           "--eval_seed", str(cfg.eval_seed),
           "--timeout", str(_cell_episode_timeout(cfg, model, task))]
    if preset:
        cmd += ["--randomization", str(preset)]
    if task.asset_pool:
        cmd += ["--asset_pool", task.asset_pool, "--asset_seed", str(task.asset_seed)]
    if task.max_steps is not None:
        cmd += ["--max_steps", str(task.max_steps)]
    if model.n_action_steps is not None:
        cmd += ["--n_action_steps", str(model.n_action_steps)]
    if cfg.clip_gripper:
        cmd += ["--clip_gripper"]
    if cfg.resume:
        cmd += ["--resume"]
    if cfg.save_video:
        cmd += ["--save_video"]
        if cfg.video_camera:
            cmd += ["--video_camera", str(cfg.video_camera)]
    for k, v in model.extra_args.items():
        cmd += [str(k), str(v)]
    return cmd


def run_cell(cfg: MatrixConfig, model: ModelSpec, task: TaskEntry, preset) -> dict | None:
    """Run one matrix cell with outer force-kill + auto-resume. Returns eval_results.json."""
    cell = _cell_dir(cfg, model, task, preset)
    cell.mkdir(parents=True, exist_ok=True)
    n_ep = task.num_episodes if task.num_episodes is not None else cfg.num_episodes
    results_path = cell / "eval_results.json"
    cmd = build_cmd(cfg, model, task, preset, cell)
    timeout_per_run = _cell_timeout_per_run(cfg, model, task)
    stdout_log = (cell / "stdout.log").open("w")

    def _read() -> dict | None:
        try:
            return json.loads(results_path.read_text())
        except Exception:
            return None

    for attempt in range(cfg.max_retries + 1):
        _wait_gpu_idle(cfg.min_gpu_free_mib, cfg.gpu_wait_max_s)
        tag = f"{model.name} | {task.task} | {preset or 'base'}"
        print(f"  [cell] {tag} (attempt {attempt + 1}/{cfg.max_retries + 1})\n"
              f"         {' '.join(cmd)}\n         log: {cell / 'stdout.log'}")
        t0 = time.monotonic()
        # start_new_session so we can SIGKILL the whole group (Kit owner + workers + server)
        proc = subprocess.Popen(cmd, stdout=stdout_log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_per_run)
        except subprocess.TimeoutExpired:
            print(f"  [cell] TIMEOUT after {time.monotonic() - t0:.0f}s "
                  f"(>{timeout_per_run}s) — SIGKILL process group")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            rc = -1
            _kill_ghost_kits()

        res = _read()
        done = bool(res) and int(res.get("num_episodes", 0)) >= n_ep
        print(f"  [cell] {tag}: rc={rc} ({time.monotonic() - t0:.0f}s) "
              f"| {res.get('success_count', 0) if res else 0}/{res.get('num_episodes', 0) if res else 0} done"
              f"{' [complete]' if done else ' [partial → resume]' if attempt < cfg.max_retries else ' [giving up]'}")
        if done or not cfg.resume:
            break
    stdout_log.close()
    return _read()


def run_matrix(cfg: MatrixConfig) -> list[dict]:
    cells = [(m, t, p) for m in cfg.models for t in cfg.tasks
             for p in (t.randomizations or [None])]
    print(f"== benchmark: {len(cfg.models)} models x {len(cfg.tasks)} tasks "
          f"= {len(cells)} cells -> {cfg.output_dir} (run_id={cfg.run_id}) ==")
    rows: list[dict] = []
    for i, (model, task, preset) in enumerate(cells):
        print(f"\n-- cell {i + 1}/{len(cells)} --")
        res = run_cell(cfg, model, task, preset)
        cell = _cell_dir(cfg, model, task, preset)
        run_name = str(cell.relative_to(Path(cfg.output_dir)))
        rows.append(_row_from_results(res or {}, run_name, model=model.name, preset=preset))
        report = _report_dir(cfg)
        _write(sorted(rows, key=lambda r: (r["task"], r["model"], r["run"])), report)
        task_rows = [r for r in rows if r["task"] == task.task]
        _write(sorted(task_rows, key=lambda r: (r["preset"], r["model"], r["run"])),
               _task_run_dir(cfg, task))
    return rows


def dry_run(cfg: MatrixConfig) -> None:
    cells = [(m, t, p) for m in cfg.models for t in cfg.tasks for p in (t.randomizations or [None])]
    print(f"== DRY RUN: {len(cells)} cells -> {cfg.output_dir} (run_id={cfg.run_id}) ==")
    for model, task, preset in cells:
        print(f"  {model.name:16} | {task.task} | preset={preset or 'base'}")
        print(f"      {' '.join(build_cmd(cfg, model, task, preset, _cell_dir(cfg, model, task, preset)))}")


# ===========================================================================
# Reporting
# ===========================================================================
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (center - half), 1), round(100 * (center + half), 1))


def _row_from_results(r: dict, run_name: str, model: str | None = None, preset=None) -> dict:
    n, k = int(r.get("num_episodes", 0)), int(r.get("success_count", 0))
    lo, hi = _wilson_ci(k, n)
    return {
        "task": r.get("task", "?"),
        "preset": (preset if preset is not None else r.get("randomization")) or "base",
        "model": model or r.get("policy_type", "?"),
        "run": run_name,
        "policy_type": r.get("policy_type", "?"),
        "checkpoint": r.get("checkpoint", ""),
        "num_episodes": n, "success_count": k,
        "success_rate": round(100 * k / n, 1) if n else 0.0,
        "score": r.get("mean_score"),
        "ci95_lo": lo, "ci95_hi": hi,
    }


def _collect(eval_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in eval_dirs:
        f = d / "eval_results.json"
        if f.exists():
            rows.append(_row_from_results(json.loads(f.read_text()), d.name))
    return sorted(rows, key=lambda r: (r["task"], r["model"], r["run"]))


def _write(rows: list[dict], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "comparison.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)
    lines = ["| task | preset | model | success | rate | score | 95% CI | N |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc = r.get("score")
        sc_str = str(sc) if sc is not None else "-"
        lines.append(f"| {r['task']} | {r['preset']} | {r['model']} | "
                     f"{r['success_count']}/{r['num_episodes']} | {r['success_rate']}% | "
                     f"{sc_str} | [{r['ci95_lo']}, {r['ci95_hi']}] | {r['num_episodes']} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "comparison.md").write_text(md)
    return md


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="yaml matrix config → RUN then report")
    ap.add_argument("--dry_run", action="store_true", help="with --config: print planned cells, run nothing")
    ap.add_argument("--eval_root", default=None, help="report-only: scan every */eval_results.json under here")
    ap.add_argument("--eval_dirs", nargs="+", default=None, help="report-only: explicit eval run dirs")
    ap.add_argument("--out_dir", default=None, help="where to write comparison.* (default: config.output_dir / eval_root)")
    args = ap.parse_args()

    if args.config:
        cfg = _load_config(args.config)
        if args.out_dir:
            cfg.output_dir = args.out_dir
        if args.dry_run:
            dry_run(cfg)
            return
        rows = run_matrix(cfg)
        out = Path(args.out_dir) if args.out_dir else _report_dir(cfg)
    else:
        if args.eval_dirs:
            dirs = [Path(d) for d in args.eval_dirs]
        elif args.eval_root:
            dirs = sorted(p.parent for p in Path(args.eval_root).rglob("eval_results.json"))
        else:
            ap.error("give --config (run) or --eval_root/--eval_dirs (report-only)")
        rows = _collect(dirs)
        out = Path(args.out_dir) if args.out_dir else (
            Path(args.eval_root) if args.eval_root else dirs[0].parent)

    if not rows:
        print("no eval_results.json found")
        return
    md = _write(sorted(rows, key=lambda r: (r["task"], r["model"], r["run"])), out)
    print(md)
    print(f"-> {out}/comparison.{{md,csv,json}}")


if __name__ == "__main__":
    main()
