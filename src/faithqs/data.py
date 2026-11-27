"""Data: MeQSum (train/dev) and MEDIQA 2021 task 1 (dev/test), built once with a manifest.

Protocol (docs/PROTOCOL.md section 2):
- whitespace normalization only, medical content untouched;
- near-duplicate questions (Jaccard >= 0.8 on word 3-grams) are grouped and a group never
  straddles two splits;
- MeQSum questions that near-duplicate an evaluation question are removed from training;
- MeQSum is split 900 / 100 with seed 42; dev = 100 MeQSum + 50 MEDIQA validation;
  test = the 100 official MEDIQA 2021 test questions;
- every split is hashed in a versioned manifest and verified on load.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MEQSUM_REPO = "albertvillanova/meqsum"
MEQSUM_REVISION = "96a6a56931c1e2c2b85e97452b902cb59d8ec94e"
MEQSUM_FILE = "MeQSum.parquet"

MEDIQA_REPO = "abachaa/MEDIQA2021"
MEDIQA_COMMIT = "38f6e5f368aec4e083623907fc56f2c49cd0fc84"
MEDIQA_FILES = {
    "mediqa2021_val": "Task1/MEDIQA2021-Task1-QuestionSummarization-ValidationSet.xlsx",
    "mediqa2021_test": "Task1/MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx",
}

SPLITS = ("train", "dev", "test")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Record:
    id: str
    source: str
    target: str
    dataset: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildParams:
    seed: int = 42
    dev_size: int = 100
    jaccard_threshold: float = 0.8
    ngram: int = 3


# --------------------------------------------------------------------------- text utilities


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace; never touches the medical content."""
    return " ".join(str(text).split())


def word_ngrams(text: str, n: int = 3) -> frozenset[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return frozenset([tuple(words)]) if words else frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- grouping & split


def near_duplicate_groups(
    texts: Sequence[str], threshold: float = 0.8, n: int = 3
) -> list[list[int]]:
    """Connected components of the 'Jaccard >= threshold' graph (exact, O(N^2))."""
    grams = [word_ngrams(t, n) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(grams[i], grams[j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def cross_overlaps(
    pool: Sequence[str], references: Sequence[str], threshold: float = 0.8, n: int = 3
) -> list[tuple[int, int, float]]:
    """(pool index, reference index, jaccard) for every near-duplicate pair across two sets."""
    ref_grams = [word_ngrams(t, n) for t in references]
    pairs = []
    for i, text in enumerate(pool):
        grams = word_ngrams(text, n)
        for j, ref in enumerate(ref_grams):
            score = jaccard(grams, ref)
            if score >= threshold:
                pairs.append((i, j, round(score, 4)))
    return pairs


def group_aware_split(
    groups: Sequence[Sequence[int]], holdout_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Shuffle whole groups deterministically and fill the holdout up to ``holdout_size``."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    holdout: list[int] = []
    keep: list[int] = []
    for g in order:
        target = holdout if len(holdout) + len(groups[g]) <= holdout_size else keep
        target.extend(groups[g])
    if len(holdout) != holdout_size:
        raise ValueError(f"Could not fill a holdout of {holdout_size} with whole groups.")
    return sorted(keep), sorted(holdout)


# --------------------------------------------------------------------------- hashing & IO


def records_sha256(records: Iterable[Record]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as f:
        return [Record(**json.loads(line)) for line in f if line.strip()]


def load_split(split: str, data_dir: Path, manifest_path: Path) -> list[Record]:
    """Load a split and verify it against the manifest (ids and content hash)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["splits"][split]
    records = read_jsonl(Path(data_dir) / manifest["version"] / f"{split}.jsonl")
    if [r.id for r in records] != expected["ids"] or records_sha256(records) != expected["sha256"]:
        raise ValueError(f"Split '{split}' does not match {manifest_path}; rebuild the data.")
    return records


# --------------------------------------------------------------------------- source loading


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())
    return dest


def load_meqsum(cache_dir: Path) -> tuple[list[Record], dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            MEQSUM_REPO,
            MEQSUM_FILE,
            repo_type="dataset",
            revision=MEQSUM_REVISION,
            cache_dir=str(cache_dir),
        )
    )
    table = pq.read_table(path).to_pylist()
    records = [
        Record(
            id=f"meqsum:{row['File']}",
            source=normalize_whitespace(row["CHQ"]),
            target=normalize_whitespace(row["Summary"]),
            dataset="meqsum",
        )
        for row in table
        if row.get("CHQ") and row.get("Summary")
    ]
    source = {
        "repo": MEQSUM_REPO,
        "revision": MEQSUM_REVISION,
        "file": MEQSUM_FILE,
        "sha256": _sha256_file(path),
        "rows": len(table),
    }
    return records, source


def load_mediqa(cache_dir: Path) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    import openpyxl

    out: dict[str, list[Record]] = {}
    files: dict[str, Any] = {}
    for name, rel in MEDIQA_FILES.items():
        url = f"https://raw.githubusercontent.com/{MEDIQA_REPO}/{MEDIQA_COMMIT}/{rel}"
        path = _download(url, Path(cache_dir) / "mediqa2021" / MEDIQA_COMMIT / Path(rel).name)
        rows = list(openpyxl.load_workbook(path, read_only=True).worksheets[0].values)
        header = [str(h).strip() for h in rows[0]]
        records = []
        for values in rows[1:]:
            row = dict(zip(header, values, strict=False))
            if not row.get("NLM Question") or not row.get("Summary"):
                continue
            meta = {
                key: normalize_whitespace(row[col])
                for key, col in (("focus", "Question Focus"), ("type", "Question Type"))
                if row.get(col)
            }
            records.append(
                Record(
                    id=f"{name}:{row['q_id']}",
                    source=normalize_whitespace(row["NLM Question"]),
                    target=normalize_whitespace(row["Summary"]),
                    dataset=name,
                    meta=meta,
                )
            )
        out[name] = records
        files[name] = {"path": rel, "sha256": _sha256_file(path), "rows": len(rows) - 1}
    return out, {"repo": MEDIQA_REPO, "commit": MEDIQA_COMMIT, "files": files}


# --------------------------------------------------------------------------- build


def assemble_splits(
    meqsum: Sequence[Record],
    mediqa_val: Sequence[Record],
    mediqa_test: Sequence[Record],
    params: BuildParams,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    """Pure split logic (no IO), unit-tested."""
    evaluation = list(mediqa_val) + list(mediqa_test)
    overlaps = cross_overlaps(
        [r.source for r in meqsum], [r.source for r in evaluation], params.jaccard_threshold
    )
    removed = {i for i, _, _ in overlaps}
    pool = [r for i, r in enumerate(meqsum) if i not in removed]

    groups = near_duplicate_groups([r.source for r in pool], params.jaccard_threshold, params.ngram)
    train_idx, dev_idx = group_aware_split(groups, params.dev_size, params.seed)

    splits = {
        "train": [pool[i] for i in train_idx],
        "dev": [pool[i] for i in dev_idx] + list(mediqa_val),
        "test": list(mediqa_test),
    }
    audit = {
        "removed_from_meqsum_for_eval_overlap": [
            {"meqsum": meqsum[i].id, "eval": evaluation[j].id, "jaccard": s} for i, j, s in overlaps
        ],
        "near_duplicate_groups": [[pool[i].id for i in g] for g in groups if len(g) > 1],
    }
    return splits, audit


def build(
    data_dir: Path, manifest_path: Path, version: str = "v1", params: BuildParams | None = None
) -> dict[str, Any]:
    params = params or BuildParams()
    out_dir = Path(data_dir) / version
    if out_dir.exists() or manifest_path.exists():
        raise FileExistsError(f"{out_dir} or {manifest_path} exists; use a new version.")

    cache = Path(data_dir) / "raw"
    meqsum, meqsum_source = load_meqsum(cache)
    mediqa, mediqa_source = load_mediqa(cache)
    splits, audit = assemble_splits(
        meqsum, mediqa["mediqa2021_val"], mediqa["mediqa2021_test"], params
    )

    for name, records in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", records)

    manifest = {
        "version": version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": asdict(params),
        "sources": {"meqsum": meqsum_source, "mediqa2021": mediqa_source},
        "splits": {
            name: {
                "count": len(records),
                "by_dataset": {
                    d: sum(r.dataset == d for r in records)
                    for d in sorted({r.dataset for r in records})
                },
                "sha256": records_sha256(records),
                "ids": [r.id for r in records],
            }
            for name, records in splits.items()
        },
        "audit": audit,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
