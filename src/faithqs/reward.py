"""The jargon-preservation reward R (DESIGN section 5.2).

    R = keep  * |J(y_hat) & J*| / |J*|                 essential jargon kept
      - lost  * |J* \\ J(y_hat)|                         penalty per essential term lost
      - halluc * |J(y_hat) \\ (J(x) | J*)| / |J(y_hat)|   hallucinated terms
      + fmt   * Format(y_hat)                          one question, '?', no preamble
      - length * max(0, |y_hat| / |y| - slack)          anti-copy length penalty

J* is the jargon of the reference (expert summary), J(x) the jargon of the source
question. All functions here are pure: terms are extracted beforehand, so the reward is
unit-testable without any model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from faithqs.jargon import Extractor, Term, dedupe, match_terms, terms_match


@dataclass(frozen=True)
class RewardWeights:
    keep: float = 1.0
    lost: float = 0.25
    halluc: float = 1.0
    fmt: float = 0.5
    length: float = 1.0
    length_slack: float = 1.5
    match_threshold: float = 85.0

    def digest(self) -> str:
        """Short hash recorded in every run that uses these weights."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class RewardBreakdown:
    recall: float
    n_lost: int
    halluc_rate: float
    format_ok: float
    length_ratio: float
    total: float
    lost_terms: tuple[str, ...] = ()
    hallucinated_terms: tuple[str, ...] = ()


_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)|sure\b|summary\s*:|summari[sz]ed|question\s*:|rewritten|the question)",
    re.IGNORECASE,
)


def n_words(text: str) -> int:
    return len(text.split())


def format_ok(text: str) -> float:
    """1.0 for a single, self-contained question without preamble, else 0.0."""
    stripped = text.strip()
    if not stripped or "\n" in stripped or _PREAMBLE.search(stripped):
        return 0.0
    return 1.0 if stripped.endswith("?") and stripped.count("?") == 1 else 0.0


def compute(
    prediction: str,
    prediction_terms: Sequence[Term],
    source_terms: Sequence[Term],
    reference: str,
    reference_terms: Sequence[Term],
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    essential = dedupe(reference_terms)
    predicted = dedupe(prediction_terms)
    thr = weights.match_threshold

    if essential:
        found, missing = match_terms(predicted, essential, thr)
        recall = len(found) / len(essential)
    else:
        missing, recall = [], 1.0  # nothing essential to keep

    allowed = list(source_terms) + list(essential)
    hallucinated = [
        t for t in predicted if not any(terms_match(t.norm, a.norm, thr) for a in allowed)
    ]
    halluc_rate = len(hallucinated) / len(predicted) if predicted else 0.0

    fmt = format_ok(prediction)
    ratio = n_words(prediction) / max(1, n_words(reference))
    total = (
        weights.keep * recall
        - weights.lost * len(missing)
        - weights.halluc * halluc_rate
        + weights.fmt * fmt
        - weights.length * max(0.0, ratio - weights.length_slack)
    )
    return RewardBreakdown(
        recall=round(recall, 4),
        n_lost=len(missing),
        halluc_rate=round(halluc_rate, 4),
        format_ok=fmt,
        length_ratio=round(ratio, 4),
        total=round(total, 4),
        lost_terms=tuple(t.text for t in missing),
        hallucinated_terms=tuple(t.text for t in hallucinated),
    )


def score_batch(
    predictions: Sequence[str],
    sources: Sequence[str],
    references: Sequence[str],
    extractor: Extractor,
    weights: RewardWeights = RewardWeights(),
) -> list[RewardBreakdown]:
    """Extract terms for every text (one batched NER pass) and compute R per example."""
    n = len(predictions)
    if not (len(sources) == len(references) == n):
        raise ValueError("predictions, sources and references must have the same length")
    terms = extractor.extract_batch([*predictions, *sources, *references])
    pred_t, src_t, ref_t = terms[:n], terms[n : 2 * n], terms[2 * n :]
    return [
        compute(p, pt, st, r, rt, weights)
        for p, pt, st, r, rt in zip(predictions, pred_t, src_t, references, ref_t, strict=True)
    ]
