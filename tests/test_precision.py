"""Training precision: the environment's choice, never silently emulated."""

import pytest

torch = pytest.importorskip("torch")

from faithqs.train_sft import resolve_precision  # noqa: E402


def fake_gpu(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], name: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)


def test_cpu_trains_in_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_precision("bf16") == "fp32"


def test_bf16_on_a_gpu_that_only_emulates_it_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gpu(monkeypatch, (7, 5), "Tesla T4")
    with pytest.raises(RuntimeError, match=r"Tesla T4.*no native bf16"):
        resolve_precision("bf16")
    assert resolve_precision("fp16") == "fp16"  # an explicit fp16 environment is allowed


def test_bf16_on_ampere_or_newer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gpu(monkeypatch, (8, 9), "NVIDIA L4")
    assert resolve_precision("bf16") == "bf16"


def test_unknown_precision() -> None:
    with pytest.raises(ValueError):
        resolve_precision("int8")
