"""The jargon-preservation reward R (DESIGN section 5.2).

    R = keep  * |J(y_hat) & J*| / |J*|                 essential jargon kept
      - lost  * |J* \\ J(y_hat)|                         penalty per essential term lost
      - halluc * |J(y_hat) \\ (J(x) | J*)| / |J(y_hat)|   hallucinated terms
      + fmt   * Format(y_hat)                          one question, '?', no preamble
      - length * max(0, |y_hat| / |y| - slack)          anti-copy length penalty

J* is the jargon of the reference (expert summary), J(x) the jargon of the source
question. All functions here are pure: terms are extracted beforehand, so the reward is
unit-testable without any model.

R reads the reference, so it is a *training* signal only. Best-of-N reranking at inference
uses ``compute_reference_free`` (R_free), which never sees the reference.
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


def _score(
    prediction: str,
    prediction_terms: Sequence[Term],
    essential_terms: Sequence[Term],
    allowed_terms: Sequence[Term],
    target_words: int,
    weights: RewardWeights,
) -> RewardBreakdown:
    essential = dedupe(essential_terms)
    predicted = dedupe(prediction_terms)
    thr = weights.match_threshold

    if essential:
        found, missing = match_terms(predicted, essential, thr)
        recall = len(found) / len(essential)
    else:
        missing, recall = [], 1.0  # nothing essential to keep

    hallucinated = [
        t for t in predicted if not any(terms_match(t.norm, a.norm, thr) for a in allowed_terms)
    ]
    halluc_rate = len(hallucinated) / len(predicted) if predicted else 0.0

    fmt = format_ok(prediction)
    ratio = n_words(prediction) / max(1, target_words)
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


def compute(
    prediction: str,
    prediction_terms: Sequence[Term],
    source_terms: Sequence[Term],
    reference: str,
    reference_terms: Sequence[Term],
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """Reference-based R, for training only (DPO pairs, GRPO): needs the expert summary."""
    return _score(
        prediction,
        prediction_terms,
        essential_terms=reference_terms,
        allowed_terms=[*source_terms, *reference_terms],
        target_words=n_words(reference),
        weights=weights,
    )


# Reference-free variant for best-of-N at inference: no expert summary exists at test time,
# so the source jargon stands in for J* and a length prior (median train reference length)
# stands in for |y|. "lost" is off by default: sources hold more terms than a summary keeps.
REFERENCE_FREE_WEIGHTS = RewardWeights(lost=0.0)


def compute_reference_free(
    prediction: str,
    prediction_terms: Sequence[Term],
    source_terms: Sequence[Term],
    target_words: int,
    weights: RewardWeights = REFERENCE_FREE_WEIGHTS,
) -> RewardBreakdown:
    """Deployable R_free: usable on the test split because it never reads the reference."""
    return _score(
        prediction,
        prediction_terms,
        essential_terms=source_terms,
        allowed_terms=source_terms,
        target_words=target_words,
        weights=weights,
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
