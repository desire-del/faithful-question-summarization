"""Test lock: the MEDIQA 2021 test split is read only for the accesses listed in the protocol.

PROTOCOL section 2 allows exactly two accesses, each used once:
- ``bart_reproduction``: phase 1 anchor check, no tuning depends on it;
- ``final``: phase 5, every retained system at once.

Each access appends a line to ``reports/test_ledger.jsonl`` (versioned), so anyone can audit
how often the test set was looked at. Scoring test predictions is allowed only for files
recorded in the ledger. On ephemeral machines (Colab), ``FAITHQS_STATE_DIR`` points the ledger
to a persistent folder (Google Drive) so that re-cloning the repository cannot reset it; the
entries are then merged into the versioned ledger by scripts/ingest_artifacts.py.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faithqs.provenance import git_state

LEDGER = Path("reports/test_ledger.jsonl")
ALLOWED_PURPOSES = ("bart_reproduction", "final")
STATE_ENV = "FAITHQS_STATE_DIR"


def default_ledger() -> Path:
    """The ledger must outlive ephemeral clones (Colab): FAITHQS_STATE_DIR points to a persistent
    directory (e.g. Google Drive); otherwise the versioned reports/test_ledger.jsonl is used."""
    state = os.environ.get(STATE_ENV)
    return Path(state) / "test_ledger.jsonl" if state else LEDGER


class TestLockError(PermissionError):
    """Raised when the test split is accessed outside the protocol."""


def read_ledger(ledger: Path | None = None) -> list[dict[str, Any]]:
    ledger = ledger or default_ledger()
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


def authorize(
    split: str,
    final: bool,
    purpose: str | None,
    outputs: list[str],
    ledger: Path | None = None,
) -> dict[str, Any] | None:
    """Gate an access to a split; record it in the ledger when it is the test split."""
    if split != "test":
        return None
    ledger = ledger or default_ledger()
    if not final:
        raise TestLockError("The test split requires --final (docs/PROTOCOL.md section 2).")
    if purpose not in ALLOWED_PURPOSES:
        raise TestLockError(f"Test access purpose must be one of {ALLOWED_PURPOSES}.")
    entries = read_ledger(ledger)
    if any(e["purpose"] == purpose and e["status"] == "closed" for e in entries):
        raise TestLockError(f"The '{purpose}' test access was already used and closed.")
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": purpose,
        "status": "open",
        "outputs": outputs,
        "git": git_state(),
        "previous_accesses": len(entries),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def is_recorded(prediction_file: str, ledger: Path | None = None) -> bool:
    return any(prediction_file in e["outputs"] for e in read_ledger(ledger))


def close(purpose: str, ledger: Path | None = None) -> None:
    """Mark an access as finished: no further test generation under that purpose."""
    ledger = ledger or default_ledger()
    entries = read_ledger(ledger)
    if not any(e["purpose"] == purpose for e in entries):
        raise TestLockError(f"No '{purpose}' access to close.")
    for e in entries:
        if e["purpose"] == purpose:
            e["status"] = "closed"
    ledger.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
