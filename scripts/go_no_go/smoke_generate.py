"""Go/no-go: load a causal checkpoint, render the chat template without thinking, generate.

Checks that the pinned revision loads as a text-only causal LM, that thinking mode is
really disabled, and records latency and peak memory.

    uv run python scripts/go_no_go/smoke_generate.py --env local --model qwen35_0_8b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import QUESTIONS

from faithqs import config, modeling, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="local")
    parser.add_argument("--model", default="qwen35_0_8b")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    cfg = config.compose(args.env, args.model)
    out = args.out or Path(f"reports/go_no_go/generate_{args.model}_{args.env}.json")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok = modeling.load_tokenizer(cfg.model)
    model = modeling.load_causal_lm(cfg.model, load_in_4bit=args.load_in_4bit, device_map="auto")
    model.eval()
    load_s = time.perf_counter() - t0

    prompts = [modeling.render_prompt(tok, cfg.model, modeling.dev_messages(q)) for q in QUESTIONS]
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen_s = time.perf_counter() - t0
    completions = tok.batch_decode(
        output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
    )

    leaked_thinking = any("<think>" in c or "</think>" in c for c in completions)
    report = {
        "check": "generation_smoke",
        "decision": "no-go" if leaked_thinking else "go",
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "model_class": type(model).__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "load_seconds": round(load_s, 1),
        "generate_seconds_batch": round(gen_s, 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "thinking_leaked": leaked_thinking,
        "samples": [
            {"question": q, "output": c} for q, c in zip(QUESTIONS, completions, strict=True)
        ],
        "provenance": provenance.collect(config.config_hash(cfg)),
    }
    provenance.write_json_new(out, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"samples", "provenance"}}, indent=2
        )
    )
    for sample in report["samples"]:
        print(f"\nQ: {sample['question']}\n-> {sample['output']}")
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
