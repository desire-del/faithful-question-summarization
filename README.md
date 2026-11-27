# Faithful Medical Question Summarization

> **Status: phase 1 (data, evaluation, BART anchor).** No results yet. Research use only, not a clinical tool.

Can post-training with a reward that penalizes **lost medical jargon** teach LLMs to keep the
medical terms that matter when summarizing consumer health questions, without hallucinating or
copying, and beat the state of the art on the official **MEDIQA 2021** benchmark?

| Role | Model |
|---|---|
| Anchor (encoder-decoder, literature backbone) | BART-large |
| Flagship (pending bake-off) | Qwen3.8-27B |
| Small model (recipe development) | Qwen3.5-4B |

Methods: SFT (LoRA / full), best-of-N, DPO and GRPO with a jargon-preservation reward, evaluated with
ROUGE, BERTScore, SummaC and blinded human annotation.

## Early findings (details in [FINDINGS](docs/FINDINGS.md))
- 2 of the 100 official MEDIQA 2021 test questions appear verbatim in MeQSum, the official training set.
  We remove them from training.
- Under transformers 5, a naive SummaC-ZS re-implementation silently drifts from the reference package
  (missing ALBERT segment ids, different tokenizer conversion). Ours matches `summac==0.0.4` to 1e-6.

## Documents (French working docs)
- [Design](docs/DESIGN.md) · [Model selection (ADR-001)](docs/MODEL_SELECTION.md) · [Protocol](docs/PROTOCOL.md)
- [Roadmap](docs/ROADMAP.md) · [Findings](docs/FINDINGS.md) · [Jargon extractor](docs/EXTRACTOR.md)
- [Parking lot](docs/PARKING_LOT.md) · [v1 post-mortem](docs/V1_POSTMORTEM.md)

## Development
```bash
make install        # core + dev tools
make check          # ruff + pytest
make install-train  # GPU stack (torch, transformers, trl, peft)
uv run faithqs list
uv run faithqs info --env local --model qwen35_0_8b
```

## Pipeline
```bash
uv run faithqs data fetch                       # rebuild splits, verify against data/manifests/splits_v1.json
uv run faithqs train-sft --env colab --model bart_large --sft-config configs/sft/bart_large.yaml --lr 3e-5
uv run faithqs generate --env colab --model bart_large --split dev --checkpoint <run>/model \
    --out <run>/predictions/dev.greedy.jsonl
uv run faithqs score --predictions <run>/predictions/dev.greedy.jsonl --split dev
```
Every metric is computed offline from prediction files. The test split is locked: it needs
`--final --purpose ...`, each access is logged in `reports/test_ledger.jsonl`, and the protocol
allows two (BART reproduction, final evaluation).
