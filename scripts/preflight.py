"""Preflight before a long GPU job: environment report, then the real pipeline on a tiny model.

It fails in a few minutes, with a readable reason, on what would otherwise break a protocol
run later: no GPU, a GPU without the environment's precision (T4 without native bf16), a
torchvision built for another torch, or any training / generation / metric error. It also
downloads and caches every metric model (BERTScore, SummaC, the jargon extractor).

    python scripts/preflight.py --env colab
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from faithqs import config, selfcheck


def fail(message: str) -> None:
    print(f"\n[preflight] FAILED: {message}\n", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True, help="Environment of the protocol run (colab, ...)")
    args = parser.parse_args()

    env = selfcheck.environment()
    print(json.dumps(env, indent=2))
    cfg = config.compose(args.env)
    if cfg.env.hardware.device == "cuda" and not env["cuda_available"]:
        fail("no CUDA GPU. On Colab: Runtime > Change runtime type > L4 or A100.")
    if str(env["torchvision"]).startswith("BROKEN"):
        fail("torchvision does not match torch: pip uninstall -y torchvision torchaudio")
    if cfg.env.hardware.precision == "bf16" and not env.get("native_bf16", False):
        fail(
            f"{env.get('gpu')} has no native bf16, which the '{args.env}' environment requires. "
            "On Colab: Runtime > Change runtime type > L4 or A100."
        )

    with tempfile.TemporaryDirectory() as tmp:
        result = selfcheck.pipeline(Path(tmp), env=args.env)
    overall = result["summary"]["overall"]
    print(
        json.dumps(
            {
                "precision": result["train"]["precision"],
                "peak_gpu_gb": result["train"]["peak_gpu_gb"],
                "metrics": sorted(overall),
            },
            indent=2,
        )
    )
    print("\n[preflight] OK: training, generation and every metric ran on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
