"""Score a prediction file against a verified split; write per-example scores and a summary.

Every metric is computed offline from the prediction file, so adding a metric later means
re-scoring old runs without any GPU training. Test predictions can only be scored if their
file is recorded in the test ledger (faithqs.testlock).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faithqs import data, metrics, reward, stats, testlock
from faithqs.jargon import CANDIDATES, DEFAULT_EXTRACTOR, Extractor

ALL_METRICS = ("rouge", "bertscore", "summac", "jargon", "copy")
SUMMARY_KEYS = (
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore",
    "summac",
    "jr",
    "lt",
    "hr",
    "copy_ratio",
    "length_words",
    "format_ok",
)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["id"]: row for row in rows}


def score_file(
    predictions_path: Path,
    split: str,
    data_dir: Path = Path("data/processed"),
    manifest: Path = Path("data/manifests/splits_v1.json"),
    include: Sequence[str] = ALL_METRICS,
    device: str | None = None,
    ledger: Path | None = None,
) -> dict[str, Any]:
    if split == "test" and not testlock.is_recorded(str(predictions_path), ledger):
        raise testlock.TestLockError(
            f"{predictions_path} is not recorded in the test ledger; test predictions must be "
            "produced through an authorized access (generate --final --purpose ...)."
        )
    records = data.load_split(split, data_dir, manifest)
    preds = read_predictions(predictions_path)
    missing = [r.id for r in records if r.id not in preds]
    if missing:
        raise ValueError(f"{len(missing)} examples have no prediction (e.g. {missing[:3]}).")

    hyp = [preds[r.id]["prediction"] for r in records]
    refs = [r.target for r in records]
    srcs = [r.source for r in records]
    rows: list[dict[str, Any]] = [
        {
            "id": r.id,
            "dataset": r.dataset,
            "length_words": len(h.split()),
            "format_ok": reward.format_ok(h),
        }
        for r, h in zip(records, hyp, strict=True)
    ]
    config: dict[str, Any] = {"split": split, "metrics": list(include)}

    if "rouge" in include:
        for row, sc in zip(rows, metrics.rouge(hyp, refs), strict=True):
            row.update(sc)
    if "bertscore" in include:
        f1, config["bertscore"] = metrics.bertscore(hyp, refs, device=device)
        for row, value in zip(rows, f1, strict=True):
            row["bertscore"] = value
    if "summac" in include:
        summac = metrics.cached(f"summac:{device}", lambda: metrics.SummaCZS(device=device))
        for row, value in zip(rows, summac.score(srcs, hyp), strict=True):
            row["summac"] = value
        config["summac"] = {"model": metrics.SUMMAC_MODEL, "revision": metrics.SUMMAC_REVISION}
    if "jargon" in include:
        extractor = metrics.cached(
            f"extractor:{device}", lambda: Extractor(CANDIDATES[DEFAULT_EXTRACTOR], device=device)
        )
        for row, b in zip(rows, reward.score_batch(hyp, srcs, refs, extractor), strict=True):
            row.update(
                jr=b.recall,
                lt=1 - b.recall,
                hr=b.halluc_rate,
                n_lost=b.n_lost,
                lost_terms=list(b.lost_terms),
                hallucinated_terms=list(b.hallucinated_terms),
            )
        config["extractor"] = DEFAULT_EXTRACTOR
    if "copy" in include:
        for row, h, s in zip(rows, hyp, srcs, strict=True):
            row["copy_ratio"] = metrics.copy_ratio(h, s)

    summary = {
        "predictions": str(predictions_path),
        "config": config,
        "overall": {},
        "by_dataset": {},
    }
    datasets = sorted({row["dataset"] for row in rows})
    for key in SUMMARY_KEYS:
        if key not in rows[0]:
            continue
        summary["overall"][key] = stats.bootstrap_mean([row[key] for row in rows]).as_dict()
        if len(datasets) > 1:
            for d in datasets:
                values = [row[key] for row in rows if row["dataset"] == d]
                summary["by_dataset"].setdefault(d, {})[key] = stats.bootstrap_mean(
                    values
                ).as_dict()

    stem = predictions_path.with_suffix("")
    per_example = stem.parent.parent / "scores" / f"{stem.name}.scores.jsonl"
    summary_path = per_example.with_name(f"{stem.name}.summary.json")
    if per_example.exists() or summary_path.exists():
        raise FileExistsError(f"{per_example} exists; scores are never overwritten.")
    per_example.parent.mkdir(parents=True, exist_ok=True)
    with per_example.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["paths"] = {"per_example": str(per_example), "summary": str(summary_path)}
    return summary
