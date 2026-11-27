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


data_app = typer.Typer(no_args_is_help=True, help="Build and inspect the dataset splits.")
app.add_typer(data_app, name="data")


@data_app.command("build")
def data_build(
    data_dir: Annotated[Path, typer.Option(help="Where processed splits are written.")] = Path(
        "data/processed"
    ),
    manifest: Annotated[Path, typer.Option(help="Versioned manifest path.")] = Path(
        "data/manifests/splits_v1.json"
    ),
    version: str = "v1",
) -> None:
    """Download pinned sources, build train/dev/test once and write the manifest."""
    from faithqs import data

    result = data.build(data_dir, manifest, version=version)
    for name, split in result["splits"].items():
        typer.echo(f"{name:5s} {split['count']:4d}  {split['by_dataset']}")
    audit = result["audit"]
    typer.echo(f"removed for eval overlap: {len(audit['removed_from_meqsum_for_eval_overlap'])}")
    typer.echo(f"near-duplicate groups (size > 1): {len(audit['near_duplicate_groups'])}")


DATA_DIR = Path("data/processed")
MANIFEST = Path("data/manifests/splits_v1.json")


@data_app.command("fetch")
def data_fetch(data_dir: Path = DATA_DIR, manifest: Path = MANIFEST) -> None:
    """Rebuild the splits from pinned sources and verify them against the versioned manifest."""
    from faithqs import data

    counts = data.fetch(data_dir, manifest)
    typer.echo(f"verified against {manifest}: {counts}")


@app.command("train-sft")
def train_sft_cmd(
    env: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    sft_config: Annotated[Path, typer.Option(help="e.g. configs/sft/bart_large.yaml")],
    lr: Annotated[float, typer.Option()],
    seed: Annotated[int, typer.Option()] = 0,
    run_name: Annotated[str | None, typer.Option()] = None,
    max_steps: Annotated[int, typer.Option(help="Smoke tests only.")] = -1,
) -> None:
    """Fine-tune a model on the train split (checkpoint chosen on dev loss)."""
    from faithqs import data, train_sft

    cfg = config.compose(env, model, extra=[sft_config])
    if cfg.model.arch != "seq2seq":
        raise typer.BadParameter("Decoder SFT arrives in phase 2; phase 1 trains BART only.")
    name = run_name or f"{model}_sft_lr{lr:g}_s{seed}"
    run_dir = Path(cfg.env.paths.runs_dir) / name
    manifest = train_sft.train_seq2seq(
        cfg,
        data.load_split("train", DATA_DIR, MANIFEST),
        data.load_split("dev", DATA_DIR, MANIFEST),
        run_dir,
        lr=lr,
        seed=seed,
        data_manifest=MANIFEST,
        max_steps=max_steps,
    )
    typer.echo(f"{run_dir}: best eval loss {manifest['best_eval_loss']}")


@app.command()
def generate(
    env: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    split: Annotated[str, typer.Option(help="dev or test (test needs --final --purpose).")],
    out: Annotated[Path, typer.Option(help="Prediction file (.jsonl), never overwritten.")],
    decoding: Annotated[str, typer.Option(help="greedy, beam4 or sample8")] = "greedy",
    checkpoint: Annotated[str | None, typer.Option(help="Fine-tuned model directory.")] = None,
    adapter: Annotated[str | None, typer.Option(help="LoRA adapter directory.")] = None,
    best_of_n: Annotated[bool, typer.Option(help="Rerank sample8 with R_free.")] = False,
    final: Annotated[bool, typer.Option(help="Required for the test split.")] = False,
    purpose: Annotated[str | None, typer.Option(help="bart_reproduction or final")] = None,
    batch_size: int = 16,
    seed: int = 0,
) -> None:
    """Generate predictions for a split through the single generation path."""
    from faithqs import data, testlock
    from faithqs import generate as gen

    if out.exists():
        raise typer.BadParameter(f"{out} exists; predictions are never overwritten.")
    testlock.authorize(split, final, purpose, [str(out)])
    cfg = config.compose(env, model)
    records = data.load_split(split, DATA_DIR, MANIFEST)
    dec = gen.DECODINGS[decoding]
    model_obj, tokenizer = gen.load_for_generation(cfg.model, checkpoint, adapter)
    outputs = gen.generate_texts(
        model_obj, tokenizer, cfg.model, [r.source for r in records], dec, batch_size, seed
    )
    candidates = None
    if best_of_n:
        from faithqs.jargon import CANDIDATES, DEFAULT_EXTRACTOR, Extractor

        target = gen.median_reference_words(data.load_split("train", DATA_DIR, MANIFEST))
        extractor = Extractor(CANDIDATES[DEFAULT_EXTRACTOR])
        predictions, candidates = gen.best_of_n(
            outputs, [r.source for r in records], extractor, target
        )
    else:
        predictions = [o[0] for o in outputs]
    meta = {
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "checkpoint": checkpoint,
        "adapter": adapter,
        "split": split,
        "decoding": gen.decoding_dict(dec),
        "best_of_n": best_of_n,
        "seed": seed,
        "config_hash": config.config_hash(cfg),
    }
    gen.write_predictions(out, records, predictions, meta, candidates)
    typer.echo(f"wrote {len(predictions)} predictions to {out}")


@app.command()
def score(
    predictions: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()],
    metrics: Annotated[str, typer.Option()] = "rouge,bertscore,summac,jargon,copy",
) -> None:
    """Score a prediction file (per-example scores + bootstrap summary)."""
    from faithqs import score as scoring

    summary = scoring.score_file(predictions, split, include=tuple(metrics.split(",")))
    overall = {k: round(v["mean"], 4) for k, v in summary["overall"].items()}
    typer.echo(json.dumps(overall, indent=2))


@app.command("close-test-access")
def close_test_access(purpose: Annotated[str, typer.Option()]) -> None:
    """Close a test access in the ledger once its predictions are generated and scored."""
    from faithqs import testlock

    testlock.close(purpose)
    typer.echo(f"test access '{purpose}' closed")


@app.command("list")
def list_configs(config_dir: ConfigDir = config.DEFAULT_CONFIG_DIR) -> None:
    """List the available env and model configs."""
    for group in ("env", "model"):
        typer.echo(f"{group}: {', '.join(config.available(group, config_dir))}")


if __name__ == "__main__":
    app()
