"""Bring a Colab / cluster artifact archive back into the repository.

The archive (built by the notebooks) holds ``runs/<run>/...`` (run manifests, predictions,
scores; never model weights) and ``reports/...``. This script:
- copies ``runs/<run>/`` into ``results/<phase>/<run>/`` (refusing to overwrite);
- copies new report files into ``reports/`` (refusing to overwrite a different file);
- merges ``reports/test_ledger.jsonl`` by appending only entries not already present.

    uv run python scripts/ingest_artifacts.py bart_anchor_artifacts.zip --phase phase1_bart
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath

LEDGER = Path("reports/test_ledger.jsonl")


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in archive: {name}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--phase", required=True, help="e.g. phase1_bart")
    args = parser.parse_args()

    copied, skipped, ledger_added = 0, 0, 0
    existing = LEDGER.read_text(encoding="utf-8").splitlines() if LEDGER.exists() else []
    with zipfile.ZipFile(args.archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            member = safe_member(info.filename)
            payload = z.read(info)
            if member.parts[0] == "runs":
                if {"model", "checkpoints"} & set(member.parts):
                    raise ValueError(f"Model weights must not be ingested: {member}")
                dest = Path("results") / args.phase / Path(*member.parts[1:])
            elif str(member) == "reports/test_ledger.jsonl":
                for line in payload.decode("utf-8").splitlines():
                    if line and line not in existing:
                        json.loads(line)  # must be valid JSON
                        existing.append(line)
                        ledger_added += 1
                continue
            elif member.parts[0] == "reports":
                dest = Path(*member.parts)
            else:
                raise ValueError(f"Unexpected archive member: {member}")
            if dest.exists():
                if dest.read_bytes() != payload:
                    raise FileExistsError(f"{dest} exists with different content.")
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            copied += 1
    if ledger_added:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text("\n".join(existing) + "\n", encoding="utf-8")
    print(
        f"copied {copied} files, {skipped} identical skipped, {ledger_added} ledger entries added"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
