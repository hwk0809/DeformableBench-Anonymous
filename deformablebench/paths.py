"""Repo paths and asset resolution.

Assets live at the repo top-level ``assets/`` (separate from the ``src/`` code
tree). For now they are committed to git (few, small). When the asset set grows
large, ``asset_dir`` becomes a download-on-demand ``ensure_asset`` without
changing callers.
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


REPO_ROOT = _find_repo_root()
ASSETS_ROOT = REPO_ROOT / "assets"


def asset_dir(name: str) -> Path:
    """Return the local dir for asset ``name`` (e.g. "cloth/silk_35")."""

    path = ASSETS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"asset {name!r} not found at {path}")
    return path
