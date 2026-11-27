from pathlib import Path

import pytest

from faithqs import provenance


def test_collect_has_the_fields_a_run_manifest_needs() -> None:
    record = provenance.collect(config_hash="abc123")
    assert record["config_hash"] == "abc123"
    assert {"timestamp_utc", "git", "python", "packages"} <= record.keys()
    assert record["packages"]["faithqs"] is not None


def test_write_json_new_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "out" / "report.json"
    provenance.write_json_new(target, {"a": 1})
    assert target.exists()
    with pytest.raises(FileExistsError):
        provenance.write_json_new(target, {"a": 2})
