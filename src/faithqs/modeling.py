"""Loading pinned checkpoints and rendering prompts, shared by every script.

Heavy imports (torch, transformers) stay inside functions so the core package and CI
remain installable without the GPU stack.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

# Development prompt for phase 0 only. The evaluated prompt is versioned in phase 1.
DEV_SYSTEM_PROMPT = "You summarize consumer health questions."
DEV_USER_TEMPLATE = (
    "Rewrite the following consumer health question as one short, self-contained question. "
    "Keep the medical terms.\n\nQuestion: {question}"
)


def _dtype(name: str) -> Any:
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def chat_template_kwargs(model_cfg: DictConfig) -> dict[str, Any]:
    kwargs = model_cfg.get("chat_template_kwargs")
    return dict(OmegaConf.to_container(kwargs, resolve=True)) if kwargs else {}


def load_tokenizer(model_cfg: DictConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, revision=model_cfg.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_cfg: DictConfig, load_in_4bit: bool = False, **kwargs: Any) -> Any:
    """Load the text-only causal LM (no vision tower) at the pinned revision."""
    from transformers import AutoModelForCausalLM

    if model_cfg.arch != "causal":
        raise ValueError(f"{model_cfg.name} is not a causal LM (arch={model_cfg.arch}).")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype("bfloat16"),
        )
    return AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        revision=model_cfg.revision,
        dtype=_dtype(model_cfg.dtype),
        **kwargs,
    )


def dev_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEV_SYSTEM_PROMPT},
        {"role": "user", "content": DEV_USER_TEMPLATE.format(question=question)},
    ]


def render_prompt(tokenizer: Any, model_cfg: DictConfig, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(model_cfg),
    )
