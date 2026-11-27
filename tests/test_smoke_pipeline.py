"""End-to-end smoke test on CPU: tiny BART SFT -> generation -> scoring (CI job 'smoke')."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]


def test_tiny_bart_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rouge_score")
    from faithqs import config, data, generate, score, train_sft
    from faithqs.data import Record

    train = [
        Record(
            f"meqsum:{i}",
            f"My knee hurts after running {i} km. What should I do?",
            f"How to treat knee pain after running {i} km?",
            "meqsum",
        )
        for i in range(8)
    ]
    dev = [
        Record(f"mediqa2021_val:{i}", r.source, r.target, "mediqa2021_val")
        for i, r in enumerate(train[:4])
    ]
    data_dir, manifest = tmp_path / "processed", tmp_path / "manifest.json"
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

    cfg = config.compose(
        "local",
        "tiny_bart",
        extra=[REPO / "configs/sft/bart_large.yaml"],
        overrides=["sft.batch_size=4", "sft.max_source_tokens=64"],
        config_dir=REPO / "configs",
    )
    run_dir = tmp_path / "run"
    result = train_sft.train_seq2seq(
        cfg, train, dev, run_dir, lr=3e-5, seed=0, data_manifest=manifest, max_steps=2
    )
    assert (run_dir / "model").is_dir() and result["n_train"] == 8

    model, tok = generate.load_for_generation(cfg.model, checkpoint=str(run_dir / "model"))
    outputs = generate.generate_texts(
        model,
        tok,
        cfg.model,
        [r.source for r in dev],
        generate.Decoding(max_new_tokens=8),
        batch_size=4,
    )
    pred_path = run_dir / "predictions" / "dev.greedy.jsonl"
    generate.write_predictions(pred_path, dev, [o[0] for o in outputs], {"model": "tiny"})

    summary = score.score_file(
        pred_path, "dev", data_dir=data_dir, manifest=manifest, include=("rouge", "copy")
    )
    assert {"rougeL", "copy_ratio", "format_ok"} <= set(summary["overall"])
    with pytest.raises(FileExistsError):  # scores are never overwritten
        score.score_file(pred_path, "dev", data_dir=data_dir, manifest=manifest, include=("rouge",))
