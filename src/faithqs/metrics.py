"""Evaluation metrics computed offline from prediction files.

- ROUGE-1/2/L: ``rouge-score``, F1, Porter stemmer (as in the MEDIQA literature).
- BERTScore F1: ``bert-score`` default English configuration (roberta-large, layer 17).
- SummaC-ZS: faithful re-implementation of Laban et al. (2022) zero-shot SummaC with the
  "vitc" NLI model and sentence granularity, the configuration used by FocusMed. The
  reference package pins huggingface-hub<=0.17 and cannot be installed next to
  transformers 5; equivalence is checked by scripts/validation/check_summac.py.
- Copy ratio: share of prediction 4-grams that appear in the source (copy detector).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faithqs.data import word_ngrams

SUMMAC_MODEL = "tals/albert-xlarge-vitaminc-mnli"
SUMMAC_REVISION = "3082ba54344bd9ddada2be1c5e9b4131721d2a5d"
SUMMAC_ENTAILMENT_IDX = 0  # SUPPORTS
SUMMAC_CONTRADICTION_IDX = 1  # REFUTES
# transformers 5 converts ALBERT's spiece.model differently from transformers 4.x (e.g. around
# hyphens and parentheses), which shifts NLI scores. We ship the tokenizer.json exported by
# transformers 4.35.2, the version paired with summac==0.0.4, and check its hash on load.
SUMMAC_TOKENIZER = Path(__file__).parent / "assets" / "summac_tokenizer" / "tokenizer.json"
SUMMAC_TOKENIZER_SHA256 = "8c3588b00c231c9797029e741d4053d112fb5f31e563b1ec6c4d91186dc5f54c"


def rouge(predictions: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out = []
    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)  # (target, prediction)
        out.append({k: v.fmeasure for k, v in scores.items()})
    return out


def bertscore(
    predictions: Sequence[str], references: Sequence[str], device: str | None = None
) -> tuple[list[float], dict[str, Any]]:
    import bert_score

    _, _, f1 = bert_score.score(
        list(predictions), list(references), lang="en", device=device, verbose=False
    )
    config = {"package": "bert-score", "version": bert_score.__version__, "lang": "en"}
    return [float(x) for x in f1], config


def copy_ratio(prediction: str, source: str, n: int = 4) -> float:
    pred = word_ngrams(prediction, n)
    if not pred:
        return 0.0
    return len(pred & word_ngrams(source, n)) / len(pred)


def split_sentences(text: str) -> list[str]:
    """SummaC sentence granularity: NLTK punkt sentences longer than 10 characters."""
    import nltk

    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.tokenize.sent_tokenize(text)
    return [s for s in sentences if len(s) > 10]


class SummaCZS:
    """Zero-shot SummaC: max over source sentences, mean over summary sentences of (E - C)."""

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 20,
        max_input_length: int = 500,
        max_doc_sents: int = 100,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        digest = hashlib.sha256(SUMMAC_TOKENIZER.read_bytes()).hexdigest()
        if digest != SUMMAC_TOKENIZER_SHA256:
            raise ValueError(f"Unexpected SummaC tokenizer file: {SUMMAC_TOKENIZER}")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(SUMMAC_TOKENIZER),
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                SUMMAC_MODEL, revision=SUMMAC_REVISION
            )
            .to(self.device)
            .eval()
        )
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_doc_sents = max_doc_sents

    def image(self, source: str, summary: str) -> Any:
        import numpy as np
        import torch

        doc = split_sentences(source)[: self.max_doc_sents]
        gen = split_sentences(summary)
        if not doc or not gen:
            return np.zeros((2, 1, 1))
        pairs = [(i, j) for i in range(len(doc)) for j in range(len(gen))]
        image = np.zeros((2, len(doc), len(gen)))
        for k in range(0, len(pairs), self.batch_size):
            batch = pairs[k : k + self.batch_size]
            tokens = self.tokenizer(
                [doc[i] for i, _ in batch],
                [gen[j] for _, j in batch],
                padding=True,
                truncation="only_first",
                max_length=self.max_input_length,
                # ALBERT needs segment ids; without them NLI probabilities are wrong
                # (checked against summac==0.0.4).
                return_token_type_ids=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**tokens).logits.float(), dim=-1).cpu().numpy()
            for (i, j), p in zip(batch, probs, strict=True):
                image[0, i, j] = p[SUMMAC_ENTAILMENT_IDX]
                image[1, i, j] = p[SUMMAC_CONTRADICTION_IDX]
        return image

    def score(self, sources: Sequence[str], summaries: Sequence[str]) -> list[float]:
        scores = []
        for source, summary in zip(sources, summaries, strict=True):
            image = self.image(source, summary)
            ent = image[0].max(axis=0)
            con = image[1].max(axis=0)
            scores.append(float((ent - con).mean()))
        return scores
