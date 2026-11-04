"""Word tokenization that preserves exact character offsets.

The models are trained on CoNLL-2003, whose tokens split punctuation from words.
We reproduce that at inference with a regex that emits either a word
(``\\w+`` incl. unicode letters/digits/underscore, plus intra-word ``.`` for
abbreviations like "U.S" and "'" for contractions) or a single non-space
symbol.  ``finditer`` gives us ``(text, start, end)`` for every token, so the
app can highlight entities on the *original* string without ever rebuilding it.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# A token is a run of word chars (optionally joined by . or ' inside), or one
# non-word, non-space character (punctuation / symbol).
_TOKEN_RE = re.compile(r"[^\W_]+(?:[.'][^\W_]+)*|[^\s\w]|_", re.UNICODE)


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return ``[(token, char_start, char_end), ...]`` over ``text``."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
