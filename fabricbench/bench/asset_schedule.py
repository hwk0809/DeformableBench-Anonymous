"""Validated asset pools and reproducible per-episode evaluation schedules."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

from fabricbench.paths import REPO_ROOT


REQUIRED_ASSET_FIELDS = ("variant_id", "build_key", "artifact_digest", "path")
SCHEDULE_NAME = "asset_schedule.json"


def _document_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_asset_pool(
    path: str | Path,
    *,
    expected_task: str | None = None,
    repo_root: Path = REPO_ROOT,
    validate_artifacts: bool = True,
) -> dict[str, Any]:
    """Load a fixed asset pool and verify its identity against build manifests."""

    raw_pool_path = Path(path)
    pool_path = (
        raw_pool_path.resolve()
        if raw_pool_path.is_absolute()
        else (repo_root / raw_pool_path).resolve()
    )
    if not pool_path.is_file():
        raise FileNotFoundError(f"asset pool not found: {pool_path}")
    pool = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
    if not isinstance(pool, dict):
        raise ValueError(f"asset pool root must be a mapping: {pool_path}")
    if pool.get("schema_version") != 1:
        raise ValueError(f"{pool_path}: unsupported schema_version {pool.get('schema_version')!r}")
    for field in ("pool_id", "task", "assets"):
        if not pool.get(field):
            raise ValueError(f"{pool_path}: missing non-empty {field!r}")
    if expected_task is not None and pool["task"] != expected_task:
        raise ValueError(
            f"asset pool task {pool['task']!r} does not match eval task {expected_task!r}"
        )
    if not isinstance(pool["assets"], list):
        raise ValueError(f"{pool_path}: 'assets' must be a list")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    normalized_assets = []
    for index, raw in enumerate(pool["assets"]):
        if not isinstance(raw, dict):
            raise ValueError(f"{pool_path}: assets[{index}] must be a mapping")
        missing = [field for field in REQUIRED_ASSET_FIELDS if not raw.get(field)]
        if missing:
            raise ValueError(f"{pool_path}: assets[{index}] missing {missing}")
        asset = {field: str(raw[field]) for field in REQUIRED_ASSET_FIELDS}
        if asset["variant_id"] in seen_ids:
            raise ValueError(f"{pool_path}: duplicate variant_id {asset['variant_id']!r}")
        if asset["path"] in seen_paths:
            raise ValueError(f"{pool_path}: duplicate asset path {asset['path']!r}")
        seen_ids.add(asset["variant_id"])
        seen_paths.add(asset["path"])
        for field in ("build_key", "artifact_digest"):
            value = asset[field]
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{pool_path}: invalid {field} for {asset['variant_id']!r}")

        if validate_artifacts:
            asset_dir = _resolve_path(asset["path"], repo_root)
            manifest_path = asset_dir / "build_manifest.json"
            meta_path = asset_dir / "asset_meta.yaml"
            if not asset_dir.is_dir():
                raise FileNotFoundError(f"asset directory not found: {asset_dir}")
            if not manifest_path.is_file():
                raise FileNotFoundError(f"build manifest not found: {manifest_path}")
            if not meta_path.is_file():
                raise FileNotFoundError(f"asset metadata not found: {meta_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in ("variant_id", "build_key", "artifact_digest"):
                if manifest.get(field) != asset[field]:
                    raise ValueError(
                        f"{manifest_path}: {field} mismatch: pool={asset[field]!r}, "
                        f"manifest={manifest.get(field)!r}"
                    )
        normalized_assets.append(asset)

    canonical_pool = {
        "schema_version": 1,
        "pool_id": str(pool["pool_id"]),
        "task": str(pool["task"]),
        "mother": pool.get("mother"),
        "campaign": pool.get("campaign"),
        "assets": normalized_assets,
    }
    canonical_pool["pool_digest"] = _document_digest(canonical_pool)
    canonical_pool["source_path"] = str(pool_path)
    return canonical_pool


def build_asset_schedule(
    pool: dict[str, Any], *, asset_seed: int, start_ep: int, num_episodes: int
) -> dict[str, Any]:
    """Build balanced seeded shuffled cycles over the fixed pool."""

    if num_episodes < 0:
        raise ValueError("num_episodes must be non-negative")
    assets = pool["assets"]
    if num_episodes and not assets:
        raise ValueError("cannot schedule episodes from an empty asset pool")
    rng = random.Random(int(asset_seed))
    scheduled = []
    while len(scheduled) < num_episodes:
        cycle = list(assets)
        rng.shuffle(cycle)
        scheduled.extend(cycle)
    episodes = [
        {"episode_id": start_ep + offset, "asset": asset}
        for offset, asset in enumerate(scheduled[:num_episodes])
    ]
    schedule = {
        "schema_version": 1,
        "pool_id": pool["pool_id"],
        "pool_digest": pool["pool_digest"],
        "task": pool["task"],
        "asset_seed": int(asset_seed),
        "start_ep": int(start_ep),
        "num_episodes": int(num_episodes),
        "episodes": episodes,
    }
    schedule["schedule_digest"] = _document_digest(schedule)
    return schedule


def load_or_create_asset_schedule(
    pool_path: str | Path,
    output_dir: str | Path,
    *,
    expected_task: str,
    asset_seed: int,
    start_ep: int,
    num_episodes: int,
    resume: bool,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a pool, then atomically create or strictly reuse its schedule."""

    pool = load_asset_pool(pool_path, expected_task=expected_task, repo_root=repo_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / SCHEDULE_NAME
    expected = build_asset_schedule(
        pool, asset_seed=asset_seed, start_ep=start_ep, num_episodes=num_episodes
    )
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        for field in (
            "schema_version", "pool_id", "pool_digest", "task", "asset_seed",
            "start_ep", "num_episodes", "schedule_digest",
        ):
            if schedule.get(field) != expected[field]:
                raise ValueError(
                    f"existing {schedule_path} has incompatible {field}: "
                    f"{schedule.get(field)!r} != {expected[field]!r}"
                )
        if schedule.get("episodes") != expected["episodes"]:
            raise ValueError(f"existing {schedule_path} episode assignments are incompatible")
        return pool, schedule

    completed = any(output.glob("episode_*.json"))
    if (output / "episodes").exists():
        completed = completed or any((output / "episodes").glob("episode_*.json"))
    if resume and completed:
        raise ValueError(f"cannot resume {output}: completed episodes exist but {SCHEDULE_NAME} is missing")
    tmp = schedule_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, schedule_path)
    return pool, expected


def schedule_by_episode(schedule: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["episode_id"]): item["asset"] for item in schedule["episodes"]}
