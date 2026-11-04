"""Canonical label scheme for EntityLens AI (CoNLL-2003, BIO2 / IOB2).

CoNLL-2003 as distributed on the Hugging Face hub already uses the IOB2
(a.k.a. BIO2) convention: every entity begins with a ``B-`` tag, and ``I-``
never starts a span.  We treat this ordered list as the single source of
truth for ``label2id`` / ``id2label`` across *all four* models so that metrics
and the deployed app share one mapping.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Order matters: index == class id used by every model and checkpoint.
LABELS: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

ENTITY_TYPES: List[str] = ["PER", "ORG", "LOC", "MISC"]

LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}

NUM_LABELS: int = len(LABELS)

# Padding / ignore index used by recurrent models and the CRF mask.
PAD_LABEL_ID: int = -100  # matches HF's default ignore_index for CrossEntropy

# Human-readable descriptions for the app's entity inspector.
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "PER": "Person — names of people, including first/last names and initials.",
    "ORG": "Organization — companies, agencies, institutions, teams, groups.",
    "LOC": "Location — countries, cities, regions, rivers and other places.",
    "MISC": "Miscellaneous — nationalities, events, products, works of art and "
            "other named entities that are not person, organization or location.",
}


def split_tag(tag: str) -> Tuple[str, str]:
    """Return ``(prefix, entity_type)`` for a BIO tag.

    ``"B-PER" -> ("B", "PER")``; ``"O" -> ("O", "")``.
    """
    if tag == "O":
        return "O", ""
    if "-" not in tag:
        return tag, ""
    prefix, etype = tag.split("-", 1)
    return prefix, etype


def is_valid_transition(prev: str, curr: str) -> bool:
    """Whether ``prev -> curr`` is a legal IOB2 transition.

    The only illegal move is entering an ``I-X`` tag when the previous tag is
    not ``B-X`` or ``I-X`` (an ``I-`` may never *start* a span, nor switch
    entity type without a ``B-``).
    """
    cp, ct = split_tag(curr)
    if cp != "I":
        return True
    pp, pt = split_tag(prev)
    return pp in {"B", "I"} and pt == ct
