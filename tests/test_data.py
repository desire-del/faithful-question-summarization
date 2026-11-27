import json
from pathlib import Path

import pytest

from faithqs import data
from faithqs.data import BuildParams, Record

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "manifests" / "splits_v1.json"


def rec(i: int, source: str, dataset: str = "meqsum") -> Record:
    return Record(id=f"{dataset}:{i}", source=source, target=f"summary {i}?", dataset=dataset)


def test_normalize_whitespace_only() -> None:
    assert (
        data.normalize_whitespace("  Can I take\n\tibuprofen  100mg? ")
        == "Can I take ibuprofen 100mg?"
    )


def test_jaccard_on_word_trigrams() -> None:
    a = data.word_ngrams("is aspirin safe during pregnancy")
    assert data.jaccard(a, a) == 1.0
    assert data.jaccard(a, data.word_ngrams("what treats herpes simplex virus")) == 0.0


def test_near_duplicate_groups_are_transitive() -> None:
    base = "my mother takes warfarin for atrial fibrillation can she take turmeric for arthritis"
    texts = [base, base + " please", base + " please help", "what is plantar fasciitis treatment"]
    groups = data.near_duplicate_groups(texts, threshold=0.8)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3]]


def test_group_aware_split_is_deterministic_and_keeps_groups() -> None:
    groups = [[0, 1], [2], [3], [4, 5, 6], [7], [8], [9]]
    first = data.group_aware_split(groups, holdout_size=3, seed=42)
    assert first == data.group_aware_split(groups, holdout_size=3, seed=42)
    keep, holdout = first
    assert len(holdout) == 3 and set(keep) | set(holdout) == set(range(10))
    for group in groups:
        assert set(group) <= set(keep) or set(group) <= set(holdout)


def test_assemble_removes_train_items_that_duplicate_evaluation_questions() -> None:
    leaked = "i have chronic renal disease is magnesium silicofluoride safe for people like me"
    meqsum = [rec(i, f"question number {i} about condition {i} and drug {i}") for i in range(12)]
    meqsum.append(rec(99, leaked))
    test = [rec(1, leaked + " thanks", "mediqa2021_test")]
    val = [rec(1, "how do i treat a tapeworm infection", "mediqa2021_val")]

    splits, audit = data.assemble_splits(meqsum, val, test, BuildParams(dev_size=3))

    assert "meqsum:99" not in {r.id for r in splits["train"] + splits["dev"]}
    assert audit["removed_from_meqsum_for_eval_overlap"][0]["meqsum"] == "meqsum:99"
    assert len(splits["dev"]) == 3 + 1 and splits["test"] == test
    ids = [r.id for name in data.SPLITS for r in splits[name]]
    assert len(ids) == len(set(ids))


def test_load_split_detects_tampering(tmp_path: Path) -> None:
    records = [rec(1, "a question"), rec(2, "another question")]
    data.write_jsonl(tmp_path / "v1" / "train.jsonl", records)
    manifest = {
        "version": "v1",
        "splits": {
            "train": {"ids": [r.id for r in records], "sha256": data.records_sha256(records)}
        },
    }
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    assert data.load_split("train", tmp_path, manifest_path) == records

    data.write_jsonl(tmp_path / "v1" / "train.jsonl", [records[0], rec(2, "edited question")])
    with pytest.raises(ValueError, match="does not match"):
        data.load_split("train", tmp_path, manifest_path)


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built yet")
def test_versioned_manifest_matches_the_protocol() -> None:
    manifest = json.loads(MANIFEST.read_text())
    splits = manifest["splits"]
    ids = {name: set(splits[name]["ids"]) for name in data.SPLITS}
    assert not (ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
    assert splits["test"]["count"] == 100
    assert splits["dev"]["by_dataset"] == {"mediqa2021_val": 50, "meqsum": 100}
    assert manifest["params"] == {"seed": 42, "dev_size": 100, "jaccard_threshold": 0.8, "ngram": 3}
