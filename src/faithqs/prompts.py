"""Versioned prompts. Every prediction file records the prompt version it was produced with.

Decoder models get the chat prompt below; encoder-decoder models (BART) are fine-tuned on the
raw question, as in the MeQSum / MEDIQA literature.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM = "You summarize consumer health questions."
USER_TEMPLATE = (
    "Summarize the following consumer health question into one short, self-contained question "
    "that a medical expert could answer. Keep the medical terms.\n\nQuestion: {question}"
)


def chat_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def chat_messages_fewshot(question: str, examples: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Few-shot chat prompt (bake-off): one user/assistant turn per (question, summary)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in examples:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(question=q)})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(question=question)})
    return messages


def seq2seq_input(question: str) -> str:
    return question
