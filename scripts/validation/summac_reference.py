"""Reference SummaC-ZS scores with the original package (run in an isolated environment).

This script must not import faithqs: it runs in a separate venv where summac==0.0.4 and
its old transformers/huggingface-hub pins can be installed.

    python summac_reference.py pairs.jsonl reference_scores.json
"""

import json
import sys

from summac.model_summac import SummaCZS

with open(sys.argv[1], encoding="utf-8") as f:
    pairs = [json.loads(line) for line in f]
model = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")
scores = model.score([p["source"] for p in pairs], [p["summary"] for p in pairs])["scores"]
with open(sys.argv[2], "w") as f:
    json.dump([float(s) for s in scores], f)
print(f"{len(scores)} reference scores written")
