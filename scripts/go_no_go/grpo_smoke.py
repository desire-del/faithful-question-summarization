"""Go/no-go: run a few GRPO steps (TRL + LoRA) and measure memory and time per step.

The reward is a dummy shape reward (length close to 15 words, ends with "?"): this
checks the machinery and the cost, not the jargon reward R, which is built in phase 1.

    # Laptop (pipeline check)
    uv run python scripts/go_no_go/grpo_smoke.py --env local --model qwen35_0_8b
    # Colab (small model)
    python scripts/go_no_go/grpo_smoke.py --env colab --model qwen35_4b
    # Cluster (flagship), see scripts/cluster/go_no_go_27b.sbatch
    python scripts/go_no_go/grpo_smoke.py --env cluster --model qwen38_27b --use-vllm
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import QUESTIONS

from faithqs import config, modeling, provenance

# Planned full GRPO run: 900 prompts x 2 epochs x 8 completions (DESIGN section 5).
PLANNED_COMPLETIONS = 900 * 2 * 8


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):  # conversational format
        return completion[-1]["content"] if completion else ""
    return str(completion)


def shape_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        text = completion_text(completion).strip()
        n_words = len(text.split())
        rewards.append(-abs(n_words - 15) / 15 + (1.0 if text.endswith("?") else 0.0))
    return rewards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="local")
    parser.add_argument("--model", default="qwen35_0_8b")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    cfg = config.compose(args.env, args.model)
    out = args.out or Path(f"reports/go_no_go/grpo_{args.model}_{args.env}.json")
    run_dir = Path(cfg.env.paths.runs_dir) / "go_no_go" / f"grpo_{args.model}_{int(time.time())}"

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.durations: list[float] = []
            self._start = 0.0

        def on_step_begin(self, *_: Any, **__: Any) -> None:
            self._start = time.perf_counter()

        def on_step_end(self, *_: Any, **__: Any) -> None:
            torch.cuda.synchronize()
            self.durations.append(time.perf_counter() - self._start)

    dataset = Dataset.from_list([{"prompt": modeling.dev_messages(q)} for q in QUESTIONS])
    tokenizer = modeling.load_tokenizer(cfg.model)
    model = modeling.load_causal_lm(cfg.model, load_in_4bit=args.load_in_4bit)

    grpo_args = GRPOConfig(
        output_dir=str(run_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        chat_template_kwargs=modeling.chat_template_kwargs(cfg.model),
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        seed=0,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    timer = StepTimer()

    torch.cuda.reset_peak_memory_stats()
    error = None
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=shape_reward,
            args=grpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora,
            callbacks=[timer],
        )
        trainer.train()
        history = trainer.state.log_history
    except torch.cuda.OutOfMemoryError as exc:
        error, history = f"OOM: {exc}".splitlines()[0], []

    steady = timer.durations[1:] or timer.durations  # first step includes warm-up
    step_s = statistics.mean(steady) if steady else None
    completions_per_step = args.num_generations
    report = {
        "check": "grpo_smoke",
        "decision": "go" if error is None and len(timer.durations) == args.steps else "no-go",
        "error": error,
        "model": f"{cfg.model.name}@{cfg.model.revision}",
        "model_class": type(model).__name__,
        "settings": {k: v for k, v in vars(args).items() if k != "out"},
        "step_seconds": [round(d, 2) for d in timer.durations],
        "steady_step_seconds": round(step_s, 2) if step_s else None,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "estimated_full_run_hours": (
            round(PLANNED_COMPLETIONS * step_s / completions_per_step / 3600, 2) if step_s else None
        ),
        "train_log": [
            {
                k: v
                for k, v in row.items()
                if k in {"step", "loss", "reward", "completions/mean_length"}
            }
            for row in history
        ],
        "provenance": provenance.collect(config.config_hash(cfg)),
    }
    provenance.write_json_new(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
