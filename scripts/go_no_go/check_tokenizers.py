"""Go/no-go: do two checkpoints share the same tokenizer?

Logit-level (on-policy) distillation from the flagship to the small model requires an
identical vocabulary (ADR-001, section 7). Only tokenizer files are downloaded.

    uv run python scripts/go_no_go/check_tokenizers.py --teacher qwen38_27b --student qwen35_4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import QUESTIONS

from faithqs import config, modeling, provenance


def _sha(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def describe(model_name: str, env: str) -> dict[str, Any]:
    cfg = config.compose(env, model_name)
    tok = modeling.load_tokenizer(cfg.model)
    rendered = modeling.render_prompt(tok, cfg.model, modeling.dev_messages(QUESTIONS[0]))
    return {
        "checkpoint": f"{cfg.model.name}@{cfg.model.revision}",
        "vocab_size": len(tok.get_vocab()),
        "vocab_sha256": _sha(sorted(tok.get_vocab().items())),
        "special_tokens": tok.special_tokens_map,
        "probe_ids_sha256": _sha([tok(q)["input_ids"] for q in QUESTIONS]),
        "chat_render_sha256": _sha(rendered),
        "chat_render_has_think_tag": "<think>" in rendered,
        "_tokenizer": tok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", default="qwen38_27b")
    parser.add_argument("--student", default="qwen35_4b")
    parser.add_argument("--env", default="local")
    parser.add_argument("--out", type=Path, default=Path("reports/go_no_go/tokenizers.json"))
    args = parser.parse_args()

    teacher, student = describe(args.teacher, args.env), describe(args.student, args.env)
    t_tok, s_tok = teacher.pop("_tokenizer"), student.pop("_tokenizer")
    checks = {
        "same_vocab": teacher["vocab_sha256"] == student["vocab_sha256"],
        "same_probe_encodings": teacher["probe_ids_sha256"] == student["probe_ids_sha256"],
        "same_special_tokens": teacher["special_tokens"] == student["special_tokens"],
        "same_chat_render": teacher["chat_render_sha256"] == student["chat_render_sha256"],
    }
    if not checks["same_vocab"]:
        t_vocab, s_vocab = set(t_tok.get_vocab()), set(s_tok.get_vocab())
        checks["only_in_teacher"] = sorted(t_vocab - s_vocab)[:50]
        checks["only_in_student"] = sorted(s_vocab - t_vocab)[:50]

    report = {
        "check": "tokenizer_compatibility",
        "decision": "go" if checks["same_vocab"] and checks["same_probe_encodings"] else "no-go",
        "teacher": {"config": args.teacher, **teacher},
        "student": {"config": args.student, **student},
        "checks": checks,
        "provenance": provenance.collect(),
    }
    provenance.write_json_new(args.out, report)
    print(json.dumps({"decision": report["decision"], **checks}, indent=2))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
