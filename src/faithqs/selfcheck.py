"""Environment report and end-to-end pipeline check, run before any long job.

``pipeline`` runs the real code path on a tiny random BART and eight synthetic records:
SFT (2 steps) -> generation -> scoring. It uses the target environment's precision and, on a
GPU machine, the same metric models as the protocol. A broken environment then fails in a
couple of minutes instead of after the first long training run. Used by the CI smoke test
(CPU, cheap metrics) and by scripts/preflight.py (Colab / cluster, every metric).
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faithqs import config, data, generate, score, train_sft
from faithqs.data import Record


def environment() -> dict[str, Any]:
    """Versions and GPU facts that decide whether a protocol run can start."""
    import torch
    import transformers

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        info["native_bf16"] = props.major >= 8
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    except Exception as exc:  # built for another torch: breaks transformers imports later
        info["torchvision"] = f"BROKEN ({type(exc).__name__}: {exc})"
    return info


def _tiny_splits(workdir: Path) -> tuple[list[Record], list[Record], Path, Path]:
    train = [
        Record(
            f"meqsum:{i}",
            f"My knee hurts after running {i} km. Should I take ibuprofen or see a doctor?",
            f"How to treat knee pain after running {i} km?",
            "meqsum",
        )
        for i in range(8)
    ]
    dev = [
        Record(f"mediqa2021_val:{i}", r.source, r.target, "mediqa2021_val")
        for i, r in enumerate(train[:4])
    ]
    data_dir, manifest = workdir / "processed", workdir / "manifest.json"
    splits = {"train": train, "dev": dev}
    for name, records in splits.items():
        data.write_jsonl(data_dir / "v1" / f"{name}.jsonl", records)
    manifest.write_text(
        json.dumps(
            {
                "version": "v1",
                "splits": {
                    n: {"ids": [r.id for r in rs], "sha256": data.records_sha256(rs)}
                    for n, rs in splits.items()
                },
            }
        )
    )
    return train, dev, data_dir, manifest


def pipeline(
    workdir: Path,
    env: str = "local",
    include: Sequence[str] = score.ALL_METRICS,
    config_dir: Path = config.DEFAULT_CONFIG_DIR,
) -> dict[str, Any]:
    """Tiny SFT -> generation -> scoring under ``env``; returns the paths and the summary."""
    train, dev, data_dir, manifest = _tiny_splits(workdir)
    cfg = config.compose(
        env,
        "tiny_bart",
        extra=[config_dir / "sft" / "bart_large.yaml"],
        overrides=["sft.batch_size=4", "sft.max_source_tokens=64"],
        config_dir=config_dir,
    )
    run_dir = workdir / "run"
    trained = train_sft.train_seq2seq(
        cfg, train, dev, run_dir, lr=3e-5, seed=0, data_manifest=manifest, max_steps=2
    )
    model, tok = generate.load_for_generation(cfg.model, checkpoint=str(run_dir / "model"))
    outputs = generate.generate_texts(
        model,
        tok,
        cfg.model,
        [r.source for r in dev],
        generate.Decoding(max_new_tokens=8),
        batch_size=4,
    )
    predictions = run_dir / "predictions" / "dev.greedy.jsonl"
    generate.write_predictions(predictions, dev, [o[0] for o in outputs], {"model": "tiny"})
    summary = score.score_file(
        predictions, "dev", data_dir=data_dir, manifest=manifest, include=include
    )
    return {
        "run_dir": run_dir,
        "predictions": predictions,
        "data_dir": data_dir,
        "manifest": manifest,
        "train": trained,
        "summary": summary,
    }
