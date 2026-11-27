"""Medical jargon extraction and matching, shared by the reward R and the metrics.

A *term* is a span of medical vocabulary (problem, drug, anatomy, procedure...) or of
critical context (dose, age, duration, pregnancy). Matching is on the **surface form**
(lower-cased, punctuation-stripped), tolerant to patient typos ("diabeties"), because the
goal is to keep the jargon itself, not only the concept (DESIGN section 5.1).

The NER backend is chosen empirically on the CHQA gold annotations
(scripts/validate_extractor.py); this module stays backend-agnostic.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

JARGON = "jargon"
CONTEXT = "context"


@dataclass(frozen=True)
class Term:
    text: str  # surface form as found
    norm: str  # normalized form used for matching
    kind: str  # JARGON or CONTEXT
    label: str  # backend label (e.g. Disease_disorder, Dosage, rule:pregnancy)
    start: int = -1
    end: int = -1


# --------------------------------------------------------------------------- normalization

_PUNCT = re.compile(r"[^\w\s/.%-]+", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_term(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    text = _SPACES.sub(" ", text).strip(" .-/")
    # crude plural folding: "tears" ~ "tear", but keep short words and -ss/-us/-is endings
    words = [
        w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w
        for w in text.split()
    ]
    return " ".join(words)


# Prefixes with opposite clinical meaning: a fuzzy match must never bridge them
# ("hypothyroidism" vs "hyperthyroidism" scores ~89 on character similarity).
_CONTRASTIVE_PREFIXES = (("hypo", "hyper"), ("brady", "tachy"), ("micro", "macro"))
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MIN_FUZZY_LENGTH = 5


def _contrastive_clash(a: str, b: str) -> bool:
    for p, q in _CONTRASTIVE_PREFIXES:
        for x, y in ((a, b), (b, a)):
            if re.search(rf"\b{p}", x) and re.search(rf"\b{q}", y):
                return True
    return False


def terms_match(a: str, b: str, threshold: float = 85.0) -> bool:
    """Surface match between two normalized terms, tolerant to typos and containment.

    - identical, or
    - one term is a whole-word sub-span of the other and the shorter has >= 4 characters
      ("type 2 diabetes" ~ "diabetes"), or
    - fuzzy ratio >= threshold for typos ("diabeties" ~ "diabetes"), only if both terms have
      >= 5 characters, carry exactly the same numbers ("50 mcg" != "500 mcg",
      "type 1" != "type 2") and no opposite prefixes ("hypo-" != "hyper-").
    """
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long_):
        return True
    if min(len(a), len(b)) < MIN_FUZZY_LENGTH:
        return False
    if _NUMBER.findall(a) != _NUMBER.findall(b) or _contrastive_clash(a, b):
        return False
    return fuzz.ratio(a, b) >= threshold


def match_terms(
    candidates: Sequence[Term], references: Sequence[Term], threshold: float = 85.0
) -> tuple[list[Term], list[Term]]:
    """Split ``references`` into (found in candidates, missing from candidates)."""
    found, missing = [], []
    for ref in references:
        if any(terms_match(ref.norm, c.norm, threshold) for c in candidates):
            found.append(ref)
        else:
            missing.append(ref)
    return found, missing


def dedupe(terms: Iterable[Term]) -> list[Term]:
    """Keep the first occurrence of each normalized form (and drop sub-spans of kept terms)."""
    kept: list[Term] = []
    for term in sorted(terms, key=lambda t: (-len(t.norm), t.start)):
        if not term.norm:
            continue
        if any(term.norm == k.norm or terms_match(term.norm, k.norm, 101.0) for k in kept):
            continue  # exact duplicate or whole-word sub-span of a longer kept term
        kept.append(term)
    return sorted(kept, key=lambda t: t.start)


# --------------------------------------------------------------------------- rules (context)

_NUM = r"\d+(?:[.,]\d+)?"
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "rule:dose",
        re.compile(
            rf"\b{_NUM}\s?(?:mg|mcg|µg|ug|g|ml|mL|iu|IU|units?|mmol|meq|%)\b", re.IGNORECASE
        ),
    ),
    ("rule:temperature", re.compile(rf"\b{_NUM}\s?(?:°\s?)?(?:f|c)\b", re.IGNORECASE)),
    (
        "rule:age",
        re.compile(
            rf"\b{_NUM}[\s-]?(?:years?|yrs?|y/o|yo|months?|weeks?)[\s-]?old\b|\baged?\s{_NUM}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rule:duration",
        re.compile(rf"\b(?:for\s)?{_NUM}\s(?:days?|weeks?|months?|years?)\b", re.IGNORECASE),
    ),
    ("rule:pregnancy", re.compile(r"\bpregnan\w*|\bbreast[- ]?feeding\b|\bnursing\b", re.I)),
]


def rule_terms(text: str) -> list[Term]:
    terms = []
    for label, pattern in _RULES:
        for m in pattern.finditer(text):
            terms.append(Term(m.group(0), normalize_term(m.group(0)), CONTEXT, label, *m.span()))
    return terms


# --------------------------------------------------------------------------- NER backend


@dataclass(frozen=True)
class NERSpec:
    """A Hugging Face token-classification model and how its labels map to term kinds."""

    model: str
    revision: str
    jargon_labels: frozenset[str]
    context_labels: frozenset[str] = frozenset()
    min_score: float = 0.5


# Candidates evaluated on CHQA (see docs/EXTRACTOR.md for the comparison and the choice).
# Labels are compared upper-cased (backends differ in casing).
MACCROBAT_JARGON = frozenset(
    {
        "DISEASE_DISORDER",
        "SIGN_SYMPTOM",
        "MEDICATION",
        "DIAGNOSTIC_PROCEDURE",
        "THERAPEUTIC_PROCEDURE",
        "BIOLOGICAL_STRUCTURE",
    }
)
MACCROBAT_CONTEXT = frozenset({"DOSAGE", "AGE", "SEX", "DURATION", "LAB_VALUE", "FREQUENCY"})

CANDIDATES: dict[str, list[NERSpec]] = {
    "deberta_maccrobat": [
        NERSpec(
            "blaze999/Medical-NER",
            "a9d5061193e969de80b24225f926cb224caac1ce",
            MACCROBAT_JARGON,
            MACCROBAT_CONTEXT,
        )
    ],
    "distilbert_maccrobat": [
        NERSpec(
            "d4data/biomedical-ner-all",
            "015a4050c9ac99722e61c547aa9b4282bcbedc7f",
            MACCROBAT_JARGON,
            MACCROBAT_CONTEXT,
        )
    ],
    "openmed_trio": [
        NERSpec(
            "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
            "da6257021de179e8479c6b336f8bd274b525646f",
            frozenset({"DISEASE"}),
        ),
        NERSpec(
            "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M",
            "d4259d05b0e924ad786f5c342ddedf46fe956331",
            frozenset({"CHEM"}),
        ),
        NERSpec(
            "OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-434M",
            "8e175ed62aebfb3d418781cd5768e5ba1357400f",
            frozenset({"ANATOMY"}),
        ),
    ],
}
# Complementary ensemble: OpenMed is strong on problems/drugs/anatomy but misses procedures
# and tests, where the MACCROBAT DistilBERT is strong (CHQA selection half, phase 1).
CANDIDATES["openmed_plus_procedures"] = [
    *CANDIDATES["openmed_trio"],
    NERSpec(
        "d4data/biomedical-ner-all",
        "015a4050c9ac99722e61c547aa9b4282bcbedc7f",
        frozenset({"DIAGNOSTIC_PROCEDURE", "THERAPEUTIC_PROCEDURE"}),
        MACCROBAT_CONTEXT,
    ),
]


# Chosen on CHQA gold annotations (docs/EXTRACTOR.md, reports/extractor/validation_v2.json).
DEFAULT_EXTRACTOR = "openmed_plus_procedures"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip whitespace and edge punctuation that tokenizers attach to entity offsets."""
    while start < end and (text[start].isspace() or text[start] in ",.;:()[]\"'"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",.;:()[]\"'?!"):
        end -= 1
    return start, end


def _merge_adjacent(text: str, terms: list[Term]) -> list[Term]:
    """Join same-kind spans separated only by nothing, a space or a hyphen inside one word
    group (e.g. "stiff" + "-person syndrome"); recompute the normalized form."""
    merged: list[Term] = []
    for term in sorted(terms, key=lambda t: (t.start, -t.end)):
        prev = merged[-1] if merged else None
        gap = text[prev.end : term.start] if prev else None
        if prev and prev.kind == term.kind and term.start >= prev.end and gap in ("", "-"):
            merged[-1] = Term(
                text[prev.start : term.end], "", prev.kind, prev.label, prev.start, term.end
            )
        else:
            merged.append(term)
    return [Term(t.text, normalize_term(t.text), t.kind, t.label, t.start, t.end) for t in merged]


class Extractor:
    """NER backend(s) + context rules -> deduplicated terms."""

    def __init__(
        self,
        specs: Sequence[NERSpec],
        device: int | str | None = None,
        batch_size: int = 32,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.specs = list(specs)
        self.batch_size = batch_size
        if pipeline_factory is None:
            from transformers import pipeline as pipeline_factory
        self._pipes = [
            pipeline_factory(
                "token-classification",
                model=spec.model,
                revision=spec.revision,
                # word-level aggregation: a word is never split across entities
                # ("simple" produced fragments such as "def | rillator")
                aggregation_strategy="first",
                stride=64,  # long patient e-mails are chunked instead of truncated
                device=device,
            )
            for spec in self.specs
        ]

    def _ner_terms(self, texts: Sequence[str]) -> list[list[Term]]:
        out: list[list[Term]] = [[] for _ in texts]
        for spec, pipe in zip(self.specs, self._pipes, strict=True):
            for i, entities in enumerate(pipe(list(texts), batch_size=self.batch_size)):
                for ent in entities:
                    label = str(ent.get("entity_group", ent.get("entity", ""))).upper()
                    if ent["score"] < spec.min_score:
                        continue
                    kind = (
                        JARGON
                        if label in spec.jargon_labels
                        else CONTEXT
                        if label in spec.context_labels
                        else None
                    )
                    if kind is None:
                        continue
                    start, end = _trim(texts[i], ent["start"], ent["end"])
                    if end - start >= 2:
                        out[i].append(Term(texts[i][start:end], "", kind, label, start, end))
        return [_merge_adjacent(text, terms) for text, terms in zip(texts, out, strict=True)]

    def extract_batch(self, texts: Sequence[str]) -> list[list[Term]]:
        ner = self._ner_terms(texts)
        return [dedupe(n + rule_terms(t)) for n, t in zip(ner, texts, strict=True)]

    def extract(self, text: str) -> list[Term]:
        return self.extract_batch([text])[0]
