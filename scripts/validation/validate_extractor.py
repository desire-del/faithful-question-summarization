"""Choose the jargon extractor on gold annotations (CHQA corpus, Kilicoglu et al. 2018).

CHQA-email holds 1,740 real consumer health e-mails sent to the NLM (the same source as
MeQSum) with expert entity annotations, typos included. We compare NER backends on
lenient span overlap with the gold *jargon* entities, choose on one half (seed 0) and
report on the other half.

    uv run python scripts/validation/validate_extractor.py \
        --candidates deberta_maccrobat distilbert_maccrobat openmed_trio
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

from faithqs import jargon, provenance

CHQA_URL = "https://bionlp.nlm.nih.gov/CHIQAcollections/CHQA-Corpus-1.0.zip"
GOLD_JARGON = {
    "Problem",
    "DrugSupplement",
    "Substance",
    "Anatomy",
    "DiagnosticProcedure",
    "ProcedureDevice",
    "GeneProtein",
    "CellularEntity",
}
GOLD_CONTEXT = {"Measurement"}


def fetch_chqa(root: Path) -> tuple[Path, str]:
    archive = root / "CHQA-Corpus-1.0.zip"
    if not archive.exists():
        root.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CHQA_URL, timeout=120) as r:
            archive.write_bytes(r.read())
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    out = root / "CHQA-Corpus-1.0"
    if not out.exists():
        zipfile.ZipFile(io.BytesIO(archive.read_bytes())).extractall(root)
    return out / "CHQA-email" / "1720_Unadjudicated", sha


def load_gold(base: Path) -> list[dict]:
    docs = []
    for ann in sorted((base / "MACE").glob("*.ann")):
        txt = base / "TextFiles" / (ann.stem + ".txt")
        if not txt.exists():
            continue
        text = txt.read_text(errors="ignore")
        spans = []
        for line in ann.read_text(errors="ignore").splitlines():
            if not line.startswith("T"):
                continue
            _, info, surface = [*line.split("\t"), "", ""][:3]
            label, *offsets = info.split()
            if ";" in " ".join(offsets):  # discontinuous span: keep the outer bounds
                offsets = " ".join(offsets).replace(";", " ").split()
            start, end = int(offsets[0]), int(offsets[-1])
            spans.append({"label": label, "start": start, "end": end, "text": surface})
        docs.append({"id": ann.stem, "text": text, "spans": spans})
    return docs


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score(
    docs: list[dict], preds: list[list[jargon.Term]], kind: str, gold_labels: set[str]
) -> dict:
    tp_p = n_p = tp_g = n_g = 0
    per_label: dict[str, list[int]] = {}
    for doc, terms in zip(docs, preds, strict=True):
        gold = [s for s in doc["spans"] if s["label"] in gold_labels]
        pred = [t for t in terms if t.kind == kind and t.start >= 0]
        n_p += len(pred)
        n_g += len(gold)
        tp_p += sum(
            any(overlaps((t.start, t.end), (g["start"], g["end"])) for g in gold) for t in pred
        )
        for g in gold:
            hit = any(overlaps((t.start, t.end), (g["start"], g["end"])) for t in pred)
            tp_g += hit
            per_label.setdefault(g["label"], [0, 0])
            per_label[g["label"]][0] += hit
            per_label[g["label"]][1] += 1
    p = tp_p / n_p if n_p else 0.0
    r = tp_g / n_g if n_g else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "n_pred": n_p,
        "n_gold": n_g,
        "recall_by_label": {k: round(v[0] / v[1], 3) for k, v in sorted(per_label.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", nargs="+", default=list(jargon.CANDIDATES))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chqa"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("reports/extractor/validation.json"))
    parser.add_argument("--previous", default=None, help="Earlier report to merge results from.")
    args = parser.parse_args()

    base, archive_sha = fetch_chqa(args.raw_dir)
    docs = load_gold(base)
    order = list(range(len(docs)))
    random.Random(0).shuffle(order)
    halves = {
        "selection": [docs[i] for i in order[: len(order) // 2]],
        "report": [docs[i] for i in order[len(order) // 2 :]],
    }

    results = {}
    for name in args.candidates:
        extractor = jargon.Extractor(jargon.CANDIDATES[name], device=args.device)
        results[name] = {}
        for half, subset in halves.items():
            t0 = time.perf_counter()
            preds = extractor.extract_batch([d["text"] for d in subset])
            elapsed = time.perf_counter() - t0
            results[name][half] = {
                "jargon": score(subset, preds, jargon.JARGON, GOLD_JARGON),
                "context_vs_measurement": score(subset, preds, jargon.CONTEXT, GOLD_CONTEXT),
                "seconds_per_100_docs": round(100 * elapsed / len(subset), 2),
            }
        print(
            name,
            json.dumps(
                {
                    h: results[name][h]["jargon"]
                    | {"s/100": results[name][h]["seconds_per_100_docs"]}
                    for h in halves
                },
                indent=1,
            ),
        )
        del extractor

    previous = Path(args.previous) if args.previous else None
    if previous and previous.exists():  # earlier candidates stay in the comparison
        for name, res in json.loads(previous.read_text())["results"].items():
            results.setdefault(name, res)
    chosen = max(results, key=lambda n: results[n]["selection"]["jargon"]["f1"])
    report = {
        "check": "extractor_selection",
        "rule": "highest lenient-overlap jargon F1 on the selection half, "
        "reported on the other half",
        "chosen": chosen,
        "corpus": {
            "url": CHQA_URL,
            "sha256": archive_sha,
            "adjudication": "MACE",
            "documents": len(docs),
            "gold_jargon_labels": sorted(GOLD_JARGON),
        },
        "results": results,
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(args.out, report)
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
