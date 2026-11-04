"""Load real pretrained GloVe embeddings and align them to our word vocab.

Downloads GloVe 6B (Wikipedia + Gigaword) once into ``data/raw`` and builds an
embedding matrix for the recurrent models' vocabulary.  Words present in GloVe
get their pretrained vector; words absent get a small random vector (and are
still covered morphologically by the char-CNN).  The matrix is cached as
``.npy`` so later runs and Kaggle skip the parse.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_ALT_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def ensure_glove(raw_dir: str | Path, dim: int = 100) -> Path:
    """Ensure ``glove.6B.<dim>d.txt`` exists in ``raw_dir``; download if needed."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt = raw_dir / f"glove.6B.{dim}d.txt"
    if txt.exists():
        return txt

    zip_path = raw_dir / "glove.6B.zip"
    if not zip_path.exists():
        _download(GLOVE_URL, zip_path, fallback=GLOVE_ALT_URL)
    with zipfile.ZipFile(zip_path) as zf:
        member = f"glove.6B.{dim}d.txt"
        zf.extract(member, raw_dir)
    if not txt.exists():  # pragma: no cover
        raise FileNotFoundError(f"{member} not found in {zip_path}")
    return txt


def build_embedding_matrix(vocab_itos, raw_dir: str | Path, dim: int = 100,
                           cache_dir: str | Path | None = None,
                           seed: int = 42) -> Tuple[np.ndarray, dict]:
    """Return ``(matrix[V, dim], stats)`` aligned to ``vocab_itos`` order."""
    cache_dir = Path(cache_dir) if cache_dir else Path(raw_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(("|".join(vocab_itos) + f"|{dim}").encode()).hexdigest()[:12]
    cache = cache_dir / f"emb_{dim}d_{key}.npy"
    if cache.exists():
        matrix = np.load(cache)
        return matrix, {"cached": True, "shape": list(matrix.shape)}

    glove_path = ensure_glove(raw_dir, dim)
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

    rng = np.random.default_rng(seed)
    scale = 0.6 / np.sqrt(dim)
    matrix = rng.normal(0.0, scale, size=(len(vocab_itos), dim)).astype(np.float32)
    matrix[0] = 0.0  # <pad> row stays zero
    hits = 0
    for i, word in enumerate(vocab_itos):
        if i == 0:
            continue
        vec = glove.get(word)
        if vec is None and word != word.lower():
            vec = glove.get(word.lower())
        if vec is not None:
            matrix[i] = vec
            hits += 1
    np.save(cache, matrix)
    stats = {"cached": False, "vocab_size": len(vocab_itos), "glove_hits": hits,
             "coverage": round(hits / max(1, len(vocab_itos) - 1), 4), "dim": dim}
    return matrix, stats


def _download(url: str, dest: Path, fallback: str | None = None) -> None:
    import urllib.request
    for candidate in [url, fallback]:
        if candidate is None:
            continue
        try:
            print(f"Downloading GloVe from {candidate} ...")
            urllib.request.urlretrieve(candidate, dest)
            return
        except Exception as e:  # pragma: no cover - network dependent
            print(f"  failed: {e}")
    raise RuntimeError("Could not download GloVe from any source.")
