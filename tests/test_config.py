from pathlib import Path

import pytest
from omegaconf import DictConfig
from omegaconf.errors import ReadonlyConfigError

from faithqs import config

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "env").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "env" / "local.yaml").write_text("name: local\npaths:\n  runs_dir: runs\n")
    (tmp_path / "model" / "tiny.yaml").write_text("name: org/tiny\narch: causal\nlr: 1.0e-4\n")
    return tmp_path


def test_compose_places_layers_under_their_group(config_dir: Path) -> None:
    cfg = config.compose("local", "tiny", config_dir=config_dir)
    assert cfg.env.name == "local"
    assert cfg.model.name == "org/tiny"


def test_extra_files_and_overrides_win(config_dir: Path, tmp_path: Path) -> None:
    extra = tmp_path / "exp.yaml"
    extra.write_text("model:\n  lr: 5.0e-5\ntrain:\n  epochs: 3\n")
    cfg = config.compose(
        "local", "tiny", extra=[extra], overrides=["train.epochs=5"], config_dir=config_dir
    )
    assert cfg.model.lr == pytest.approx(5e-5)
    assert cfg.train.epochs == 5


def test_composed_config_is_read_only(config_dir: Path) -> None:
    cfg = config.compose("local", config_dir=config_dir)
    with pytest.raises(ReadonlyConfigError):
        cfg.env.name = "other"


def test_missing_layer_raises(config_dir: Path) -> None:
    with pytest.raises(config.ConfigError, match="not found"):
        config.compose("nowhere", config_dir=config_dir)


def test_hash_is_stable_and_order_independent(config_dir: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("x: 1\ny: 2\n")
    b.write_text("y: 2\nx: 1\n")
    hash_a = config.config_hash(config.compose("local", extra=[a], config_dir=config_dir))
    hash_b = config.config_hash(config.compose("local", extra=[b], config_dir=config_dir))
    assert hash_a == hash_b
    assert len(hash_a) == 12


@pytest.mark.parametrize("model", config.available("model", REPO_CONFIGS))
def test_repository_model_configs_are_complete(model: str) -> None:
    cfg = config.compose("local", model, config_dir=REPO_CONFIGS)
    assert isinstance(cfg.model, DictConfig)
    assert cfg.model.arch in {"causal", "seq2seq"}
    # Every model is pinned to an exact Hub commit for reproducibility.
    assert len(cfg.model.revision) == 40


@pytest.mark.parametrize("env", ["local", "colab", "cluster"])
def test_repository_env_configs_resolve(env: str) -> None:
    cfg = config.compose(env, config_dir=REPO_CONFIGS)
    assert cfg.env.name == env
    assert cfg.env.generation_backend in {"hf", "vllm"}
