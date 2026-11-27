"""Flagship bake-off (ADR-001 section 5, PROTOCOL section 6). Inference only, dev split only.

generate  few-shot (k=5, the same training examples for every model) greedy predictions
decide    score JR - HR per example and apply the pre-registered rule:
          best mean JR - HR wins; if its 95% CI overlaps the runner-up's, keep Qwen3.8-27B;
          ROUGE-L only breaks exact ties.

# cluster (80 GB GPU), one model per call
python scripts/bakeoff.py generate --env cluster --model qwen38_27b
python scripts/bakeoff.py generate --env cluster --model gemma4_31b
python scripts/bakeoff.py generate --env cluster --model medgemma_27b
python scripts/bakeoff.py decide --env cluster
# Colab A100 fallback: add --load-in-4bit (the bias is reported)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from faithqs import config, data, generate, provenance, score, stats

DATA_DIR = Path("data/processed")
MANIFEST = Path("data/manifests/splits_v1.json")
OUT = Path("reports/bakeoff")
CANDIDATES = ("qwen38_27b", "gemma4_31b", "medgemma_27b")
DEFAULT = "qwen38_27b"
K, FEWSHOT_SEED = 5, 0


def fewshot_examples() -> list[tuple[str, str, str]]:
    train = data.load_split("train", DATA_DIR, MANIFEST)
    return [(r.id, r.source, r.target) for r in random.Random(FEWSHOT_SEED).sample(train, K)]


def run_generate(env: str, model: str, load_in_4bit: bool) -> None:
    cfg = config.compose(env, model)
    runs = Path(cfg.env.paths.runs_dir) / "bakeoff" / model
    path = runs / "predictions" / ("dev.fewshot5" + (".4bit" if load_in_4bit else "") + ".jsonl")
    dev = data.load_split("dev", DATA_DIR, MANIFEST)
    shots = fewshot_examples()
    model_obj, tok = generate.load_for_generation(cfg.model, load_in_4bit=load_in_4bit)
    dec = generate.DECODINGS["greedy"]
    outputs = generate.generate_texts(
        model_obj,
        tok,
        cfg.model,
        [r.source for r in dev],
        dec,
        batch_size=8,
        fewshot=[(q, a) for _, q, a in shots],
    )
    meta = {
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "split": "dev",
        "decoding": generate.decoding_dict(dec),
        "fewshot_ids": [i for i, _, _ in shots],
        "load_in_4bit": load_in_4bit,
        "config_hash": config.config_hash(cfg),
    }
    generate.write_predictions(path, dev, [o[0] for o in outputs], meta)
    score.score_file(path, "dev", DATA_DIR, MANIFEST, include=("rouge", "jargon", "copy"))
    print(f"{model}: predictions and scores under {runs}")


def run_decide(env: str) -> None:
    cfg = config.compose(env)
    base = Path(cfg.env.paths.runs_dir) / "bakeoff"
    per_model, rouge = {}, {}
    for model in CANDIDATES:
        files = sorted((base / model / "scores").glob("dev.fewshot5*.scores.jsonl"))
        if not files:
            raise FileNotFoundError(f"No bake-off scores for {model}; run 'generate' first.")
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        per_model[model] = [r["jr"] - r["hr"] for r in rows]
        rouge[model] = sum(r["rougeL"] for r in rows) / len(rows)

    estimates = {m: stats.bootstrap_mean(v) for m, v in per_model.items()}
    ranked = sorted(CANDIDATES, key=lambda m: (estimates[m].mean, rouge[m]), reverse=True)
    best, second = ranked[0], ranked[1]
    overlap = estimates[best].low <= estimates[second].high
    winner = DEFAULT if overlap else best
    small = {
        "qwen38_27b": "qwen35_4b",
        "gemma4_31b": "gemma 4 E4B",
        "medgemma_27b": "medgemma 1.5 4B",
    }[winner]
    report = {
        "rule": "max mean(JR - HR) on dev; if its 95% CI overlaps the runner-up's, keep "
        "Qwen3.8-27B; ROUGE-L breaks exact ties (PROTOCOL section 6)",
        "jr_minus_hr": {m: e.as_dict() for m, e in estimates.items()},
        "rougeL": rouge,
        "ranking": ranked,
        "ci_overlap_best_vs_second": overlap,
        "flagship": winner,
        "small_model": small,
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(OUT / "decision.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["generate", "decide"])
    parser.add_argument("--env", required=True)
    parser.add_argument("--model", choices=CANDIDATES)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if args.stage == "generate":
        if not args.model:
            parser.error("--model is required for 'generate'")
        run_generate(args.env, args.model, args.load_in_4bit)
    else:
        run_decide(args.env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
