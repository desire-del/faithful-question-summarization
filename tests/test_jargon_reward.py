"""Jargon matching and adversarial tests for the reward R (no model needed)."""

import re
from collections.abc import Sequence

import pytest

from faithqs import jargon, reward
from faithqs.jargon import JARGON, Term

LEXICON = {
    "levothyroxine": "MEDICATION",
    "synthroid": "MEDICATION",
    "hypothyroidism": "DISEASE_DISORDER",
    "thyroid": "BIOLOGICAL_STRUCTURE",
    "ibuprofen": "MEDICATION",
    "fever": "SIGN_SYMPTOM",
    "metformin": "MEDICATION",
    "diabetes": "DISEASE_DISORDER",
    "diabeties": "DISEASE_DISORDER",
    "type 2 diabetes": "DISEASE_DISORDER",
}


class FakeExtractor:
    """Lexicon-based stand-in for the NER backend, plus the real context rules."""

    def extract_batch(self, texts: Sequence[str]) -> list[list[Term]]:
        out = []
        for text in texts:
            terms = []
            for word, label in LEXICON.items():
                for m in re.finditer(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                    terms.append(
                        Term(
                            m.group(0), jargon.normalize_term(m.group(0)), JARGON, label, *m.span()
                        )
                    )
            out.append(jargon.dedupe(terms + jargon.rule_terms(text)))
        return out


SOURCE = (
    "Hi, I'm 34 and 3 months pregnant. I was diagnosed with hypothyroidism last year and take "
    "levothyroxine 50 mcg every morning. Can I keep taking it during the pregnancy? Thanks"
)
REFERENCE = "Is levothyroxine safe during pregnancy for hypothyroidism?"


def r(prediction: str, **weights: float) -> reward.RewardBreakdown:
    return reward.score_batch(
        [prediction], [SOURCE], [REFERENCE], FakeExtractor(), reward.RewardWeights(**weights)
    )[0]


# ----------------------------------------------------------------------------- matching


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("diabetes", "diabetes", True),
        ("diabeties", "diabetes", True),  # patient typo
        ("type 2 diabetes", "diabetes", True),  # whole-word containment
        ("meniscus tears", "meniscus tear", True),  # plural folding happens in normalize
        ("levothyroxine", "synthroid", False),  # different drug names never match
        ("ear", "fear", False),  # short terms never match fuzzily
        ("hypothyroidism", "hyperthyroidism", False),  # opposite prefixes, ratio ~89
        ("hypoglycemia", "hyperglycemia", False),
        ("bradycardia", "tachycardia", False),
        ("50 mcg", "500 mcg", False),  # doses must be identical
        ("type 1 diabetes", "type 2 diabetes", False),
        ("levothyroxin", "levothyroxine", True),  # typo on a long term
    ],
)
def test_terms_match(a: str, b: str, expected: bool) -> None:
    assert jargon.terms_match(jargon.normalize_term(a), jargon.normalize_term(b)) is expected


def test_context_rules_catch_doses_ages_and_pregnancy() -> None:
    labels = {t.label for t in jargon.rule_terms(SOURCE)}
    assert {"rule:dose", "rule:pregnancy"} <= labels
    assert {"rule:age"} <= {t.label for t in jargon.rule_terms("my son is 4 years old")}


# ----------------------------------------------------------------------------- reward R


def test_faithful_summary_gets_the_highest_reward() -> None:
    best = r(REFERENCE)
    assert best.recall == 1.0 and best.n_lost == 0 and best.halluc_rate == 0.0
    assert best.format_ok == 1.0


def test_paraphrased_jargon_is_penalized() -> None:
    paraphrase = r("Is my thyroid medicine safe during pregnancy?")
    assert paraphrase.n_lost == 2  # levothyroxine and hypothyroidism lost
    assert paraphrase.total < r(REFERENCE).total


def test_lost_critical_context_is_penalized() -> None:
    no_pregnancy = r("Is levothyroxine safe for hypothyroidism?")
    assert "pregnancy" in [t.lower() for t in no_pregnancy.lost_terms]
    assert no_pregnancy.total < r(REFERENCE).total


def test_hallucinated_drug_is_penalized() -> None:
    swapped = r("Is Synthroid safe during pregnancy for hypothyroidism?")
    assert swapped.hallucinated_terms == ("Synthroid",)
    assert swapped.total < r(REFERENCE).total


def test_copying_the_source_is_penalized_by_length_and_format() -> None:
    copied = r(SOURCE)
    assert copied.recall == 1.0  # copying keeps every term...
    assert copied.format_ok == 0.0 and copied.length_ratio > 3
    assert copied.total < r(REFERENCE).total  # ...but must not win


def test_without_length_penalty_copying_is_no_longer_punished() -> None:
    # The H3b ablation relies on this: removing the length term opens the copy loophole.
    assert r(SOURCE, length=0.0).total > r(SOURCE).total


def test_preamble_and_empty_outputs_fail_the_format() -> None:
    assert r("Here is the summary: Is levothyroxine safe during pregnancy?").format_ok == 0.0
    empty = r("")
    assert empty.recall == 0.0 and empty.format_ok == 0.0
    assert empty.total < r(REFERENCE).total


def test_weights_digest_changes_with_weights() -> None:
    assert reward.RewardWeights().digest() != reward.RewardWeights(lost=0.5).digest()
    assert len(reward.RewardWeights().digest()) == 12


def test_reference_free_reward_never_reads_the_reference() -> None:
    ext = FakeExtractor()
    pred = "Is levothyroxine safe during pregnancy for hypothyroidism?"
    pred_t, src_t = ext.extract_batch([pred, SOURCE])
    free = reward.compute_reference_free(pred, pred_t, src_t, target_words=11)
    # same result whatever the reference would have been: the function has no access to it
    assert free.halluc_rate == 0.0 and free.format_ok == 1.0
    swapped = "Is Synthroid safe during pregnancy?"
    sw_t = ext.extract_batch([swapped])[0]
    assert reward.compute_reference_free(swapped, sw_t, src_t, target_words=11).total < free.total


def test_reference_free_reward_still_punishes_copying() -> None:
    ext = FakeExtractor()
    good, copy = "Is levothyroxine safe during pregnancy for hypothyroidism?", SOURCE
    g_t, c_t, s_t = ext.extract_batch([good, copy, SOURCE])
    assert (
        reward.compute_reference_free(copy, c_t, s_t, 11).total
        < reward.compute_reference_free(good, g_t, s_t, 11).total
    )
