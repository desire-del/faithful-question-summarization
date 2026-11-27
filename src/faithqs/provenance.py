"""Run provenance: what code, config and environment produced an artifact."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("faithqs", "torch", "transformers", "trl", "peft", "accelerate", "vllm")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "count": torch.cuda.device_count(),
        "memory_gb": round(props.total_memory / 1024**3, 2),
        "cuda": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported(),
    }


def collect(config_hash: str | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "config_hash": config_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }


def write_json_new(path: Path, data: dict[str, Any]) -> Path:
    """Write JSON, refusing to overwrite an existing artifact."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new output path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path
