"""End-to-end smoke test on CPU: tiny BART SFT -> generation -> scoring (CI job 'smoke')."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]


def test_tiny_bart_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rouge_score")
    from faithqs import score, selfcheck

    result = selfcheck.pipeline(
        tmp_path, env="local", include=("rouge", "copy"), config_dir=REPO / "configs"
    )
    assert (result["run_dir"] / "model").is_dir() and result["train"]["n_train"] == 8
    assert {"rougeL", "copy_ratio", "format_ok"} <= set(result["summary"]["overall"])
    with pytest.raises(FileExistsError):  # scores are never overwritten
        score.score_file(
            result["predictions"],
            "dev",
            data_dir=result["data_dir"],
            manifest=result["manifest"],
            include=("rouge",),
        )
