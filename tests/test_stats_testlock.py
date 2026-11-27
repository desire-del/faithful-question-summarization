from pathlib import Path

import pytest

from faithqs import stats, testlock


def test_kendall_tau_b_extremes_and_ties() -> None:
    assert stats.kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Reference values computed with scipy.stats.kendalltau (tau-b), scipy not a dependency.
    assert stats.kendall_tau_b([1, 1, 2, 3], [1, 2, 2, 3]) == pytest.approx(0.8)
    x, y = [3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8]
    assert stats.kendall_tau_b(x, y) == pytest.approx(0.16051447, abs=1e-7)


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1, 0.4, 0.35, 0.8, 0.5, 0.2, 0.9, 0.3]
    first = stats.bootstrap_mean(values, n_resamples=2000)
    assert first == stats.bootstrap_mean(values, n_resamples=2000)
    assert first.low < first.mean < first.high


def test_paired_difference_detects_a_consistent_gain() -> None:
    base = [0.30, 0.42, 0.25, 0.51, 0.33, 0.47, 0.29, 0.38, 0.44, 0.36]
    better = [v + 0.05 for v in base]
    diff = stats.paired_bootstrap_diff(better, base, n_resamples=2000)
    assert diff.mean == pytest.approx(0.05) and diff.excludes_zero()


def test_mcnemar_exact_matches_binomial_reference() -> None:
    # b=10, c=2: two-sided exact p = 2 * P(X <= 2 | n=12, p=0.5) = 0.03857...
    assert stats.mcnemar_exact(10, 2) == pytest.approx(0.0385742, abs=1e-6)
    assert stats.mcnemar_exact(0, 0) == 1.0


def test_test_split_requires_final_and_an_allowed_purpose(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    assert testlock.authorize("dev", final=False, purpose=None, outputs=[], ledger=ledger) is None
    with pytest.raises(testlock.TestLockError, match="--final"):
        testlock.authorize("test", final=False, purpose="final", outputs=[], ledger=ledger)
    with pytest.raises(testlock.TestLockError, match="purpose"):
        testlock.authorize("test", final=True, purpose="peek", outputs=[], ledger=ledger)


def test_each_access_is_logged_and_usable_once(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    testlock.authorize("test", True, "bart_reproduction", ["runs/bart/test.jsonl"], ledger)
    assert testlock.is_recorded("runs/bart/test.jsonl", ledger)
    assert not testlock.is_recorded("runs/other/test.jsonl", ledger)
    testlock.close("bart_reproduction", ledger)
    with pytest.raises(testlock.TestLockError, match="already used"):
        testlock.authorize("test", True, "bart_reproduction", ["x"], ledger)
    assert len(testlock.read_ledger(ledger)) == 1


def test_ledger_follows_the_persistent_state_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(testlock.STATE_ENV, raising=False)
    assert testlock.default_ledger() == testlock.LEDGER
    monkeypatch.setenv(testlock.STATE_ENV, str(tmp_path / "drive"))
    assert testlock.default_ledger() == tmp_path / "drive" / "test_ledger.jsonl"
    testlock.authorize("test", True, "final", ["p.jsonl"])  # no explicit ledger
    assert testlock.is_recorded("p.jsonl")
    assert (tmp_path / "drive" / "test_ledger.jsonl").exists()
