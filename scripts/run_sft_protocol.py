"""SFT protocol (docs/PROTOCOL.md section 3), stage by stage.

    search  one run per learning rate (seed 0), dev greedy predictions, choice on dev ROUGE-L
    seeds   remaining seeds at the chosen rate; dev predictions (greedy + beam4) and scores
    table   mean +/- std across seeds on dev (overall and per dataset)
    test    ONE logged test access for the BART reproduction: every seed, greedy + beam4,
            compared with FocusMed's BART (MEDIQA 2021 test)

    python scripts/run_sft_protocol.py --env colab --model bart_large \
        --sft-config configs/sft/bart_large.yaml --tag bart --stage search
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from faithqs import config, data, generate, provenance, score, testlock, train_sft

DATA_DIR = Path("data/processed")
MANIFEST = Path("data/manifests/splits_v1.json")
# Protocol state (lr selection) must survive ephemeral clones: with FAITHQS_STATE_DIR set
# (Google Drive on Colab) it lives there, next to the test ledger.
_STATE = os.environ.get(testlock.STATE_ENV)
REPORTS = Path(_STATE) / "reports" / "sft" if _STATE else Path("reports/sft")
TABLE_KEYS = (
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore",
    "summac",
    "jr",
    "hr",
    "copy_ratio",
    "length_words",
    "format_ok",
)
# FocusMed (arXiv 2510.04671), table V: BART re-run on the MEDIQA 2021 test set.
FOCUSMED_BART = {"rougeL": 0.293, "bertscore": 0.747, "summac": 0.448}


def run_name(tag: str, lr: float, seed: int) -> str:
    return f"{tag}_lr{lr:g}_s{seed}"


LEDGER: Path | None = None  # None -> testlock.default_ledger() (persistent state dir if set)


def fail(message: str) -> None:
    print(f"\n[protocol] {message}\n", file=sys.stderr)
    raise SystemExit(2)


def predict_and_score(cfg, run_dir: Path, split: str, decoding: str, records) -> dict:
    model, tok = generate.load_for_generation(cfg.model, checkpoint=str(run_dir / "model"))
    dec = generate.DECODINGS[decoding]
    outputs = generate.generate_texts(model, tok, cfg.model, [r.source for r in records], dec)
    path = run_dir / "predictions" / f"{split}.{decoding}.jsonl"
    meta = {
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "checkpoint": str(run_dir / "model"),
        "split": split,
        "decoding": generate.decoding_dict(dec),
    }
    generate.write_predictions(path, records, [o[0] for o in outputs], meta)
    return score.score_file(path, split, DATA_DIR, MANIFEST, ledger=LEDGER)


def train_one(cfg, tag: str, lr: float, seed: int) -> Path:
    """Train, or reuse a finished run (the protocol can be resumed after an interruption)."""
    run_dir = Path(cfg.env.paths.runs_dir) / run_name(tag, lr, seed)
    if (run_dir / "model").exists():
        print(f"[protocol] reusing finished run {run_dir}")
        return run_dir
    if run_dir.exists():  # interrupted run: keep it aside (never delete), then retrain
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        aside = run_dir.with_name(f"{run_dir.name}.incomplete-{stamp}")
        run_dir.rename(aside)
        print(f"[protocol] {run_dir} has no saved model (interrupted run): moved to {aside}")
    train_sft.train_seq2seq(
        cfg,
        data.load_split("train", DATA_DIR, MANIFEST),
        data.load_split("dev", DATA_DIR, MANIFEST),
        run_dir,
        lr=lr,
        seed=seed,
        data_manifest=MANIFEST,
    )
    return run_dir


def dev_summary(cfg, run_dir: Path, decoding: str, dev) -> dict:
    """Score dev predictions once; reuse existing predictions / scores when resuming."""
    summary_path = run_dir / "scores" / f"dev.{decoding}.summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    predictions = run_dir / "predictions" / f"dev.{decoding}.jsonl"
    if predictions.exists():
        return score.score_file(predictions, "dev", DATA_DIR, MANIFEST)
    return predict_and_score(cfg, run_dir, "dev", decoding, dev)


def stage_search(cfg, tag: str) -> None:
    selection = REPORTS / f"{tag}_lr_selection.json"
    if selection.exists():
        print(f"[protocol] search already done: {selection} (chosen lr {chosen_lr(tag):g})")
        return
    dev = data.load_split("dev", DATA_DIR, MANIFEST)
    results = {}
    for lr in cfg.sft.lr_grid:
        run_dir = train_one(cfg, tag, lr, seed=0)
        summary = dev_summary(cfg, run_dir, "greedy", dev)
        results[f"{lr:g}"] = summary["overall"]["rougeL"]["mean"]
        print(f"lr={lr:g}  dev ROUGE-L={results[f'{lr:g}']:.4f}")
    chosen = max(cfg.sft.lr_grid, key=lambda lr: (results[f"{lr:g}"], -lr))  # tie -> lower lr
    out = {
        "tag": tag,
        "rule": "max dev ROUGE-L (greedy, seed 0); ties -> lower lr",
        "dev_rougeL": results,
        "chosen_lr": chosen,
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(REPORTS / f"{tag}_lr_selection.json", out)
    print(f"chosen lr: {chosen:g}")


def chosen_lr(tag: str) -> float:
    selection = REPORTS / f"{tag}_lr_selection.json"
    if not selection.exists():
        fail(
            f"{selection} is missing: run '--stage search' first and check that it ends with "
            "'chosen lr: ...'. If the clone was reset, set FAITHQS_STATE_DIR to a persistent "
            "folder (Google Drive on Colab)."
        )
    return json.loads(selection.read_text())["chosen_lr"]


def stage_seeds(cfg, tag: str, seeds: list[int]) -> None:
    lr = chosen_lr(tag)
    dev = data.load_split("dev", DATA_DIR, MANIFEST)
    for seed in seeds:
        run_dir = train_one(cfg, tag, lr, seed)
        for decoding in ("greedy", "beam4"):
            dev_summary(cfg, run_dir, decoding, dev)


def aggregate(cfg, tag: str, seeds: list[int], split: str, decoding: str) -> dict:
    lr = chosen_lr(tag)
    summaries = [
        json.loads(
            (
                Path(cfg.env.paths.runs_dir)
                / run_name(tag, lr, s)
                / "scores"
                / f"{split}.{decoding}.summary.json"
            ).read_text()
        )
        for s in seeds
    ]

    def agg(section: list[dict]) -> dict:
        return {
            k: {
                "mean": statistics.fmean(s[k]["mean"] for s in section),
                "std": statistics.stdev(s[k]["mean"] for s in section) if len(section) > 1 else 0.0,
            }
            for k in TABLE_KEYS
            if k in section[0]
        }

    table = {"overall": agg([s["overall"] for s in summaries]), "by_dataset": {}}
    for d in summaries[0]["by_dataset"]:
        table["by_dataset"][d] = agg([s["by_dataset"][d] for s in summaries])
    return table


def markdown(table: dict, title: str) -> str:
    cols = [k for k in TABLE_KEYS if k in table["overall"]]
    lines = [
        f"### {title}",
        "",
        "| subset | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, row in [("all", table["overall"]), *table["by_dataset"].items()]:
        cells = [f"{row[c]['mean']:.3f} ± {row[c]['std']:.3f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def stage_table(cfg, tag: str, seeds: list[int]) -> None:
    out = {d: aggregate(cfg, tag, seeds, "dev", d) for d in ("greedy", "beam4")}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{tag}_dev.json").write_text(json.dumps(out, indent=2) + "\n")
    md = "".join(
        markdown(out[d], f"{tag} - dev - {d} (mean ± std over seeds {seeds})") + "\n" for d in out
    )
    (REPORTS / f"{tag}_dev.md").write_text(md)
    print(md)


def stage_test(cfg, tag: str, seeds: list[int]) -> None:
    lr = chosen_lr(tag)
    runs = [Path(cfg.env.paths.runs_dir) / run_name(tag, lr, s) for s in seeds]
    missing = [str(r) for r in runs if not (r / "model").exists()]
    if missing:  # checked BEFORE the access is logged
        fail(
            f"No trained model for {missing}: run '--stage seeds' first. The test was not accessed."
        )
    outputs = [
        str(r / "predictions" / f"test.{d}.jsonl") for r in runs for d in ("greedy", "beam4")
    ]
    testlock.authorize(
        "test", final=True, purpose="bart_reproduction", outputs=outputs, ledger=LEDGER
    )
    test = data.load_split("test", DATA_DIR, MANIFEST)
    for run_dir in runs:
        for decoding in ("greedy", "beam4"):
            predict_and_score(cfg, run_dir, "test", decoding, test)
    testlock.close("bart_reproduction", ledger=LEDGER)
    result = {d: aggregate(cfg, tag, seeds, "test", d)["overall"] for d in ("greedy", "beam4")}
    comparison = {
        d: {
            k: {
                "ours_mean": result[d][k]["mean"],
                "ours_std": result[d][k]["std"],
                "focusmed_bart": v,
                "diff": result[d][k]["mean"] - v,
            }
            for k, v in FOCUSMED_BART.items()
        }
        for d in result
    }
    report = {
        "purpose": "bart_reproduction",
        "seeds": seeds,
        "lr": lr,
        "test": result,
        "vs_focusmed_bart": comparison,
        "dod": "ROUGE-L within +/-0.02 of 0.293 (ROADMAP phase 1)",
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(REPORTS / f"{tag}_test_reproduction.json", report)
    print(json.dumps(comparison, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-config", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", required=True, choices=["search", "seeds", "table", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--override", nargs="*", default=[], help="Config dotlist overrides.")
    parser.add_argument(
        "--dry-run",
        type=Path,
        default=None,
        help="Pipeline check: reports and test ledger go to this scratch directory.",
    )
    args = parser.parse_args()
    if args.dry_run:
        global REPORTS, LEDGER
        REPORTS, LEDGER = args.dry_run / "reports", args.dry_run / "test_ledger.jsonl"
    cfg = config.compose(args.env, args.model, extra=[args.sft_config], overrides=args.override)
    if args.stage == "search":
        stage_search(cfg, args.tag)
    elif args.stage == "seeds":
        stage_seeds(cfg, args.tag, args.seeds)
    elif args.stage == "table":
        stage_table(cfg, args.tag, args.seeds)
    else:
        stage_test(cfg, args.tag, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
