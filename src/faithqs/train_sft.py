"""Supervised fine-tuning. Phase 1: encoder-decoder (BART) with Seq2SeqTrainer.

Checkpoint selection on dev loss (early stopping); the learning rate is chosen on dev ROUGE-L
by the caller. Each run writes ``run_manifest.json`` (config, seed, lr, data manifest hash,
provenance) next to the saved model and never reuses an existing directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from faithqs import config, prompts, provenance
from faithqs.data import Record


def _manifest_digest(manifest: Path) -> str:
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def train_seq2seq(
    cfg: DictConfig,
    train: Sequence[Record],
    dev: Sequence[Record],
    run_dir: Path,
    lr: float,
    seed: int,
    data_manifest: Path,
    max_steps: int = -1,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; every run gets a new directory.")
    run_dir.mkdir(parents=True)
    sft = cfg.sft
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name, revision=cfg.model.revision)

    def encode(batch: dict[str, list[str]]) -> dict[str, Any]:
        enc = tokenizer(
            [prompts.seq2seq_input(s) for s in batch["source"]],
            max_length=sft.max_source_tokens,
            truncation=True,
        )
        enc["labels"] = tokenizer(
            text_target=batch["target"], max_length=sft.max_target_tokens, truncation=True
        )["input_ids"]
        return enc

    def to_dataset(records: Sequence[Record]) -> Any:
        ds = Dataset.from_list([{"source": r.source, "target": r.target} for r in records])
        return ds.map(encode, batched=True, remove_columns=["source", "target"])

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=lr,
        num_train_epochs=sft.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=sft.batch_size,
        per_device_eval_batch_size=sft.batch_size,
        gradient_accumulation_steps=sft.grad_accum,
        warmup_steps=float(sft.warmup_ratio),  # transformers 5: a float < 1 is a ratio
        weight_decay=sft.weight_decay,
        eval_strategy="epoch" if max_steps < 0 else "steps",
        save_strategy="epoch" if max_steps < 0 else "steps",
        eval_steps=None if max_steps < 0 else max_steps,
        save_steps=None if max_steps < 0 else max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        bf16=bf16,
        logging_steps=10,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(dev),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=sft.early_stopping_patience)],
    )
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0

    model_dir = run_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    for ckpt in (run_dir / "checkpoints").glob("checkpoint-*"):  # keep only the best model
        import shutil

        shutil.rmtree(ckpt)

    manifest = {
        "kind": "sft_seq2seq",
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "lr": lr,
        "seed": seed,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint_step": trainer.state.best_model_checkpoint,
        "epochs_run": trainer.state.epoch,
        "train_seconds": round(seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available()
        else None,
        "n_train": len(train),
        "n_dev": len(dev),
        "data_manifest_sha256": _manifest_digest(data_manifest),
        "config": config.to_dict(cfg),
        "log_history": trainer.state.log_history,
        "provenance": provenance.collect(config.config_hash(cfg)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
