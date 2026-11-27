# Faithful Medical Question Summarization

> **Status: phase 0 (scaffolding).** No results yet. Research use only, not a clinical tool.

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

## Documents (French working docs)
- [Design](docs/DESIGN.md) · [Model selection (ADR-001)](docs/MODEL_SELECTION.md) · [Protocol](docs/PROTOCOL.md)
- [Roadmap](docs/ROADMAP.md) · [Parking lot](docs/PARKING_LOT.md) · [v1 post-mortem](docs/V1_POSTMORTEM.md)

## Development
```bash
make install        # core + dev tools
make check          # ruff + pytest
make install-train  # GPU stack (torch, transformers, trl, peft)
uv run faithqs list
uv run faithqs info --env local --model qwen35_0_8b
```
