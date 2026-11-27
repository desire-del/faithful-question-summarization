"""Command-line entry point: ``faithqs <command>``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf

from faithqs import config, provenance

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigDir = Annotated[Path, typer.Option(help="Directory holding env/ and model/ configs.")]


@app.command()
def info(
    env: Annotated[str, typer.Option(help="Environment config: local, colab or cluster.")],
    model: Annotated[str | None, typer.Option(help="Model config name.")] = None,
    override: Annotated[list[str] | None, typer.Option(help="Dotlist override.")] = None,
    config_dir: ConfigDir = config.DEFAULT_CONFIG_DIR,
) -> None:
    """Print the resolved configuration, its hash and the runtime provenance."""
    cfg = config.compose(env, model, overrides=override or [], config_dir=config_dir)
    typer.echo(OmegaConf.to_yaml(cfg))
    typer.echo(json.dumps(provenance.collect(config.config_hash(cfg)), indent=2))


@app.command("list")
def list_configs(config_dir: ConfigDir = config.DEFAULT_CONFIG_DIR) -> None:
    """List the available env and model configs."""
    for group in ("env", "model"):
        typer.echo(f"{group}: {', '.join(config.available(group, config_dir))}")


if __name__ == "__main__":
    app()
