"""Word- and character-level vocabularies for the recurrent models.

Both vocabularies are built from the **training split only** to avoid leaking
dev/test information.  Unknown words at inference map to ``<unk>``; OOV is
handled semantically by the character-level encoder (see ``models.char_cnn``),
so even unseen words contribute morphological features.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

PAD = "<pad>"
UNK = "<unk>"


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK])

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]


@dataclass
class Vocabularies:
    word: Vocab
    char: Vocab
    lower: bool = True
    max_word_len: int = 20
    meta: Dict = field(default_factory=dict)

    def encode_word(self, token: str) -> int:
        return self.word.encode(token.lower() if self.lower else token)

    def encode_chars(self, token: str) -> List[int]:
        ids = [self.char.encode(c) for c in token[: self.max_word_len]]
        if not ids:
            ids = [self.char.pad_id]
        return ids

    # ---- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "word_itos": self.word.itos,
            "char_itos": self.char.itos,
            "lower": self.lower,
            "max_word_len": self.max_word_len,
            "meta": self.meta,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "Vocabularies":
        d = json.loads(Path(path).read_text())
        word = Vocab(stoi={t: i for i, t in enumerate(d["word_itos"])}, itos=d["word_itos"])
        char = Vocab(stoi={t: i for i, t in enumerate(d["char_itos"])}, itos=d["char_itos"])
        return cls(word=word, char=char, lower=d["lower"],
                   max_word_len=d["max_word_len"], meta=d.get("meta", {}))


def build_vocabularies(train_examples: List[Dict], min_freq: int = 1,
                       lower: bool = True, max_word_len: int = 20) -> Vocabularies:
    word_counter: Counter = Counter()
    char_counter: Counter = Counter()
    for ex in train_examples:
        for tok in ex["tokens"]:
            word_counter[tok.lower() if lower else tok] += 1
            for c in tok:
                char_counter[c] += 1

    word_itos = [PAD, UNK] + [w for w, c in word_counter.most_common() if c >= min_freq]
    char_itos = [PAD, UNK] + [c for c, _ in char_counter.most_common()]

    word = Vocab(stoi={t: i for i, t in enumerate(word_itos)}, itos=word_itos)
    char = Vocab(stoi={t: i for i, t in enumerate(char_itos)}, itos=char_itos)
    return Vocabularies(word=word, char=char, lower=lower, max_word_len=max_word_len,
                        meta={"min_freq": min_freq, "n_words": len(word_itos),
                              "n_chars": len(char_itos)})
