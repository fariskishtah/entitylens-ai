"""Unified NER predictor for both model families.

``Predictor.load(path)`` auto-detects the backend:
  * a directory containing ``config.json`` + weights -> Transformer
  * a directory containing ``best.pt`` + ``vocab.json`` -> recurrent model

``predict(text)`` returns ``{tokens, tags, entities}`` where ``entities`` have
exact character offsets into the *original* text.  Confidence is the per-token
softmax probability of the predicted tag, averaged over a span.  It is a raw
model score, **not** a calibrated probability (documented honestly in the app).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..labels import ID2LABEL
from .spans import bio_to_spans
from .tokenizer import tokenize_with_offsets


class Predictor:
    def __init__(self, backend: str, model, extra: Dict):
        self.backend = backend
        self.model = model
        self.extra = extra
        self.device = torch.device("cpu")

    # ---- loading --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        path = Path(path)
        if (path / "config.json").exists():
            return cls._load_transformer(path)
        if (path / "best.pt").exists() and (path / "vocab.json").exists():
            return cls._load_rnn(path)
        raise FileNotFoundError(f"No recognizable model artifacts in {path}")

    @classmethod
    def _load_transformer(cls, path: Path) -> "Predictor":
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForTokenClassification.from_pretrained(str(path)).eval()
        return cls("transformer", model, {"tokenizer": tok})

    @classmethod
    def _load_rnn(cls, path: Path) -> "Predictor":
        from ..data.vocab import Vocabularies
        from ..models.factory import build_rnn_model
        ckpt = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        vocabs = Vocabularies.load(path / "vocab.json")
        model = build_rnn_model(ckpt["config"], vocabs, pretrained=None)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return cls("rnn", model, {"vocabs": vocabs, "config": ckpt["config"]})

    # ---- prediction -----------------------------------------------------
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        tokens = tokenize_with_offsets(text or "")
        if not tokens:
            return {"tokens": [], "tags": [], "entities": []}
        words = [t[0] for t in tokens]
        if self.backend == "transformer":
            tags, confs = self._predict_transformer(words)
        else:
            tags, confs = self._predict_rnn(words)
        entities = bio_to_spans(text, tokens, tags, confs)
        return {"tokens": tokens, "tags": tags, "confidences": confs, "entities": entities}

    @torch.no_grad()
    def predict_pretokenized(self, words: List[str]) -> List[str]:
        """Predict BIO tags for already-tokenized words (word-level eval)."""
        if not words:
            return []
        if self.backend == "transformer":
            return self._predict_transformer(words)[0]
        return self._predict_rnn(words)[0]

    def _predict_rnn(self, words: List[str]):
        vocabs = self.extra["vocabs"]
        word_ids = torch.tensor([[vocabs.encode_word(w) for w in words]], dtype=torch.long)
        maxw = vocabs.max_word_len
        char = torch.full((1, len(words), maxw), vocabs.char.pad_id, dtype=torch.long)
        for j, w in enumerate(words):
            cids = vocabs.encode_chars(w)[:maxw]
            char[0, j, :len(cids)] = torch.tensor(cids, dtype=torch.long)
        lengths = torch.tensor([len(words)], dtype=torch.long)
        mask = torch.ones((1, len(words)), dtype=torch.bool)
        emissions = self.model.emissions(word_ids, char, lengths)
        probs = F.softmax(emissions, dim=-1)[0]
        pred_ids = self.model.decode(word_ids, char, lengths, mask)[0]
        tags = [ID2LABEL[i] for i in pred_ids]
        confs = [float(probs[i, pred_ids[i]]) for i in range(len(words))]
        return tags, confs

    def _predict_transformer(self, words: List[str]):
        tok = self.extra["tokenizer"]
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=256)
        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1)
        word_ids = enc.word_ids(batch_index=0)
        tags, confs = [], []
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            pid = int(probs[pos].argmax())
            tags.append(ID2LABEL[pid])
            confs.append(float(probs[pos, pid]))
        # any words truncated beyond max_length default to O
        while len(tags) < len(words):
            tags.append("O"); confs.append(0.0)
        return tags, confs
