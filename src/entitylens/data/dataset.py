"""Load and freeze the CoNLL-2003 NER dataset.

We pin a *parquet* distribution of CoNLL-2003 (``tomaarsen/conll2003``) which
requires no remote-code execution, and immediately persist the official
train/validation/test splits to ``data/processed`` as parquet.  Every later
stage (audit, training, Kaggle) reads those frozen files, so the splits are
identical everywhere and the pipeline runs offline after the first download.

Each example is a dict with:
    tokens:   List[str]   whitespace-free word tokens for one sentence
    ner_tags: List[int]   BIO2 class ids aligned 1:1 with ``tokens``
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from ..labels import LABELS

# Candidates tried in order. The first is a clean parquet mirror (no scripts);
# the others are fallbacks that may require ``trust_remote_code``.
DATASET_CANDIDATES = ["tomaarsen/conll2003", "eriktks/conll2003", "conll2003"]

SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}


def _hf_label_names(ds) -> List[str]:
    return ds["train"].features["ner_tags"].feature.names


def download_and_freeze(processed_dir: str | Path) -> Dict[str, object]:
    """Download CoNLL-2003 and write the official splits to ``processed_dir``.

    Returns a provenance dict (source, splits, label order, checksums) and also
    writes it to ``dataset_provenance.json``.  Idempotent: if the parquet files
    already exist with a matching provenance file, nothing is re-downloaded.
    """
    from datasets import load_dataset  # local import keeps module import cheap

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prov_path = processed_dir / "dataset_provenance.json"

    if prov_path.exists() and all(
        (processed_dir / f).exists() for f in SPLIT_FILES.values()
    ):
        return json.loads(prov_path.read_text())

    last_err = None
    ds = source = None
    for name in DATASET_CANDIDATES:
        try:
            trust = name != "tomaarsen/conll2003"
            ds = load_dataset(name, trust_remote_code=trust) if trust else load_dataset(name)
            source = name
            break
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load any CoNLL-2003 candidate: {last_err}")

    hf_labels = _hf_label_names(ds)
    if hf_labels != LABELS:
        raise ValueError(
            f"Label order from {source} ({hf_labels}) does not match the "
            f"canonical EntityLens scheme ({LABELS}). Refusing to silently remap."
        )

    checksums = {}
    for split, fname in SPLIT_FILES.items():
        out = processed_dir / fname
        ds[split].to_parquet(str(out))
        checksums[split] = _sha256(out)

    provenance = {
        "source": source,
        "candidates_tried": DATASET_CANDIDATES,
        "splits": {s: len(ds[s]) for s in SPLIT_FILES},
        "labels": LABELS,
        "checksums_sha256": checksums,
        "note": (
            "CoNLL-2003 English NER. Annotations derive from the Reuters RCV1 "
            "corpus; the benchmark is standard for NER research. See README for "
            "license/provenance details."
        ),
    }
    prov_path.write_text(json.dumps(provenance, indent=2))
    return provenance


def load_split(processed_dir: str | Path, split: str) -> List[Dict]:
    """Load one frozen split as a list of ``{tokens, ner_tags}`` dicts."""
    import pyarrow.parquet as pq

    path = Path(processed_dir) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_and_freeze() first "
            "(python -m scripts.prepare_data)."
        )
    table = pq.read_table(path, columns=["tokens", "ner_tags"])
    cols = table.to_pydict()
    return [
        {"tokens": list(t), "ner_tags": list(map(int, tags))}
        for t, tags in zip(cols["tokens"], cols["ner_tags"])
    ]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
