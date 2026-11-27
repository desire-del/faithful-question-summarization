"""Check the jargon extractor on *summaries* (CHQA only annotates questions).

    export  write 30 dev reference summaries with the detected terms to a CSV to annotate by hand
    score   read the annotated CSV and report precision / recall of the extractor on summaries

Annotation guide (one row per summary):
- ``missed_terms``: medical terms or critical context (drug, disease, symptom, body part,
  test, procedure, dose, age, pregnancy...) present in the summary but NOT in ``detected_terms``;
- ``false_terms``: entries of ``detected_terms`` that are not medical jargon / critical context.
Separate several terms with " | ". Leave a cell empty when there is nothing to report.

    uv run python scripts/validation/summary_annotation.py export
    uv run python scripts/validation/summary_annotation.py score
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from faithqs import data, provenance
from faithqs.jargon import CANDIDATES, DEFAULT_EXTRACTOR, Extractor

CSV_PATH = Path("reports/extractor/summary_annotation.csv")
SEP = " | "


def export() -> None:
    if CSV_PATH.exists():
        raise FileExistsError(f"{CSV_PATH} exists; annotate it or move it away first.")
    dev = data.load_split("dev", Path("data/processed"), Path("data/manifests/splits_v1.json"))
    sample = random.Random(0).sample(dev, 30)
    terms = Extractor(CANDIDATES[DEFAULT_EXTRACTOR]).extract_batch([r.target for r in sample])
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "summary", "detected_terms", "missed_terms", "false_terms"])
        for record, found in zip(sample, terms, strict=True):
            writer.writerow([record.id, record.target, SEP.join(t.text for t in found), "", ""])
    print(f"wrote {CSV_PATH}: fill 'missed_terms' and 'false_terms', then run 'score'")


def _split(cell: str) -> list[str]:
    return [t.strip() for t in cell.split("|") if t.strip()]


def score() -> None:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    detected = sum(len(_split(r["detected_terms"])) for r in rows)
    false = sum(len(_split(r["false_terms"])) for r in rows)
    missed = sum(len(_split(r["missed_terms"])) for r in rows)
    true_positive = detected - false
    precision = true_positive / detected if detected else 0.0
    recall = true_positive / (true_positive + missed) if true_positive + missed else 0.0
    report = {
        "check": "extractor_on_summaries",
        "extractor": DEFAULT_EXTRACTOR,
        "n_summaries": len(rows),
        "detected": detected,
        "false_positives": false,
        "missed": missed,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(CSV_PATH.with_suffix(".result.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
