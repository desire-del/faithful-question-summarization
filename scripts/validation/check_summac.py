"""Check that faithqs.metrics.SummaCZS reproduces the reference SummaC-ZS implementation.

1) build pairs from the dev split (faithful references + mismatched references),
2) run scripts/validation/summac_reference.py in an isolated venv with summac==0.0.4,
3) compare with our implementation.

    uv run python scripts/validation/check_summac.py make-pairs
    <isolated python> scripts/validation/summac_reference.py reports/summac/pairs.jsonl \
        reports/summac/reference_scores.json
    uv run python scripts/validation/check_summac.py compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from faithqs import data, provenance
from faithqs.metrics import SummaCZS

OUT = Path("reports/summac")


def make_pairs() -> None:
    dev = data.load_split("dev", Path("data/processed"), Path("data/manifests/splits_v1.json"))
    sample = dev[:30]
    pairs = [
        {"id": r.id, "source": r.source, "summary": r.target, "kind": "reference"} for r in sample
    ]
    pairs += [
        {
            "id": r.id,
            "source": r.source,
            "summary": sample[(i + 7) % 30].target,
            "kind": "mismatched",
        }
        for i, r in enumerate(sample)
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs written to {OUT / 'pairs.jsonl'}")


def compare() -> None:
    pairs = [json.loads(line) for line in (OUT / "pairs.jsonl").open(encoding="utf-8")]
    reference = json.loads((OUT / "reference_scores.json").read_text())
    ours = SummaCZS(device="cpu").score([p["source"] for p in pairs], [p["summary"] for p in pairs])
    diffs = [abs(a - b) for a, b in zip(ours, reference, strict=True)]
    report = {
        "check": "summac_equivalence",
        "n_pairs": len(pairs),
        "max_abs_diff": round(max(diffs), 6),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
        "decision": "go" if max(diffs) < 1e-3 else "no-go",
        "mean_score": {
            kind: round(
                sum(s for s, p in zip(ours, pairs, strict=True) if p["kind"] == kind) / 30, 4
            )
            for kind in ("reference", "mismatched")
        },
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(OUT / "equivalence.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    {"make-pairs": make_pairs, "compare": compare}[sys.argv[1]]()
