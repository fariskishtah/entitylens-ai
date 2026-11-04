"""Convert per-token BIO tags (+ offsets + confidences) into entity spans.

Robust to slightly malformed sequences: an ``I-X`` with no matching open span is
treated as the start of a new span (so nothing is dropped), and a type switch
mid-span closes the previous span.  Each returned span carries the exact
character offsets and the substring taken directly from the original text.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..labels import split_tag


def bio_to_spans(text: str, tokens: List[Tuple[str, int, int]], tags: List[str],
                 confidences: List[float] | None = None) -> List[Dict]:
    spans: List[Dict] = []
    cur = None  # {"type", "start", "end", "confs": [...]}

    def close():
        nonlocal cur
        if cur is not None:
            confs = cur["confs"]
            spans.append({
                "text": text[cur["start"]:cur["end"]],
                "type": cur["type"],
                "start": cur["start"],
                "end": cur["end"],
                "confidence": round(sum(confs) / len(confs), 4) if confs else None,
            })
            cur = None

    for i, tag in enumerate(tags):
        prefix, etype = split_tag(tag)
        _, s, e = tokens[i]
        conf = confidences[i] if confidences is not None else None
        if prefix == "O":
            close()
        elif prefix == "B":
            close()
            cur = {"type": etype, "start": s, "end": e, "confs": [conf] if conf is not None else []}
        else:  # I-
            if cur is not None and cur["type"] == etype:
                cur["end"] = e
                if conf is not None:
                    cur["confs"].append(conf)
            else:  # dangling I- -> open a new span
                close()
                cur = {"type": etype, "start": s, "end": e, "confs": [conf] if conf is not None else []}
    close()
    return spans
