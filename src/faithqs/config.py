"""Configuration composition.

A run configuration is built from layers, merged in this order (later wins):

1. ``configs/env/<env>.yaml``      stored under ``cfg.env``   (paths, hardware, backends)
2. ``configs/model/<model>.yaml``  stored under ``cfg.model`` (checkpoint, revision, family)
3. extra YAML files                merged at the root          (experiment settings)
4. dotlist overrides               e.g. ``["train.lr=1e-4"]``

The same code runs locally, on Colab and on the cluster: only the ``env`` layer changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_DIR = Path("configs")


class ConfigError(ValueError):
    """Raised when a configuration layer is missing or invalid."""


def _load_yaml(path: Path) -> DictConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return cfg


def available(group: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> list[str]:
    """Names of the configs available in a group (``env``, ``model``...)."""
    return sorted(p.stem for p in (config_dir / group).glob("*.yaml"))


def compose(
    env: str,
    model: str | None = None,
    extra: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> DictConfig:
    """Merge the configuration layers and return a resolved, read-only config."""
    layers = [OmegaConf.create({"env": _load_yaml(config_dir / "env" / f"{env}.yaml")})]
    if model is not None:
        layers.append(
            OmegaConf.create({"model": _load_yaml(config_dir / "model" / f"{model}.yaml")})
        )
    layers.extend(_load_yaml(Path(p)) for p in extra)
    layers.append(OmegaConf.from_dotlist(list(overrides)))

    cfg = OmegaConf.merge(*layers)
    assert isinstance(cfg, DictConfig)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def to_dict(cfg: DictConfig) -> dict:
    """Plain resolved dictionary, suitable for JSON manifests."""
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    return data


def config_hash(cfg: DictConfig) -> str:
    """Short, stable hash of the resolved configuration (key order independent)."""
    payload = json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
