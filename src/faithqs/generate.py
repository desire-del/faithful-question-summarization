"""Generation for every system through one code path, and best-of-N reranking with R_free.

Outputs: ``<name>.jsonl`` with one ``{"id", "dataset", "prediction"}`` per line, plus a
``<name>.meta.json`` sidecar (model, checkpoint, prompt version, decoding, provenance).
Best-of-N also writes ``<name>.candidates.jsonl`` with every candidate and its R_free.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from faithqs import modeling, prompts, provenance
from faithqs.data import Record


@dataclass(frozen=True)
class Decoding:
    name: str = "greedy"
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    num_return_sequences: int = 1


DECODINGS = {
    "greedy": Decoding(),
    "beam4": Decoding(name="beam4", num_beams=4),
    "sample8": Decoding(name="sample8", do_sample=True, num_return_sequences=8),
}


def load_for_generation(
    model_cfg: DictConfig, checkpoint: str | None = None, adapter: str | None = None
) -> tuple[Any, Any]:
    """Base model at its pinned revision, a full checkpoint, or base + LoRA adapter."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if model_cfg.arch == "seq2seq":
        source = checkpoint or model_cfg.name
        kwargs = {} if checkpoint else {"revision": model_cfg.revision}
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(source, **kwargs)
    else:
        tokenizer = modeling.load_tokenizer(model_cfg)
        if checkpoint:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto")
        else:
            model = modeling.load_causal_lm(model_cfg)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def generate_texts(
    model: Any,
    tokenizer: Any,
    model_cfg: DictConfig,
    sources: Sequence[str],
    decoding: Decoding,
    batch_size: int = 16,
    seed: int = 0,
) -> list[list[str]]:
    """Return, for every source, ``num_return_sequences`` generated texts."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    causal = model_cfg.arch == "causal"
    if causal:
        tokenizer.padding_side = "left"
        inputs = [
            modeling.render_prompt(tokenizer, model_cfg, prompts.chat_messages(s)) for s in sources
        ]
    else:
        inputs = [prompts.seq2seq_input(s) for s in sources]

    outputs: list[list[str]] = []
    k = decoding.num_return_sequences
    for start in range(0, len(inputs), batch_size):
        truncation = {} if causal else {"truncation": True, "max_length": 1024}
        batch = tokenizer(
            inputs[start : start + batch_size], return_tensors="pt", padding=True, **truncation
        ).to(device)
        gen_kwargs = {
            "max_new_tokens": decoding.max_new_tokens,
            "do_sample": decoding.do_sample,
            "num_beams": decoding.num_beams,
            "num_return_sequences": k,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if decoding.do_sample:
            gen_kwargs.update(temperature=decoding.temperature, top_p=decoding.top_p)
        with torch.inference_mode():
            ids = model.generate(**batch, **gen_kwargs)
        if causal:
            ids = ids[:, batch["input_ids"].shape[1] :]
        texts = [t.strip() for t in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        outputs.extend(texts[i : i + k] for i in range(0, len(texts), k))
    return outputs


def median_reference_words(train: Sequence[Record]) -> int:
    """Length prior for R_free: median word count of the training references."""
    return round(statistics.median(len(r.target.split()) for r in train))


def best_of_n(
    candidates: Sequence[Sequence[str]],
    sources: Sequence[str],
    extractor: Any,
    target_words: int,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Pick, per example, the candidate with the highest reference-free reward R_free."""
    from faithqs import reward

    flat = [c for cands in candidates for c in cands]
    flat_terms = extractor.extract_batch(flat)
    source_terms = extractor.extract_batch(list(sources))
    chosen, details, cursor = [], [], 0
    for cands, src_terms in zip(candidates, source_terms, strict=True):
        scored = []
        for text in cands:
            breakdown = reward.compute_reference_free(
                text, flat_terms[cursor], src_terms, target_words
            )
            scored.append({"text": text, "r_free": breakdown.total})
            cursor += 1
        chosen.append(max(scored, key=lambda s: s["r_free"])["text"])
        details.append(scored)
    return chosen, details


def write_predictions(
    path: Path,
    records: Sequence[Record],
    predictions: Sequence[str],
    meta: dict[str, Any],
    candidates: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> Path:
    """Write predictions + sidecar metadata; never overwrite an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; predictions are never overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record, prediction in zip(records, predictions, strict=True):
            row = {"id": record.id, "dataset": record.dataset, "prediction": prediction}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if candidates is not None:
        with path.with_suffix(".candidates.jsonl").open("w", encoding="utf-8") as f:
            for record, cands in zip(records, candidates, strict=True):
                f.write(json.dumps({"id": record.id, "candidates": cands}, ensure_ascii=False))
                f.write("\n")
    sidecar = {
        **meta,
        "prompt_version": prompts.PROMPT_VERSION,
        "n": len(records),
        "provenance": provenance.collect(meta.get("config_hash")),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2, default=str) + "\n")
    return path


def decoding_dict(decoding: Decoding) -> dict[str, Any]:
    return asdict(decoding)
