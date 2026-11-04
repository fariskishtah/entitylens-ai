"""Pure, testable helpers for the Gradio app.

Keeping rendering/stats/CSV logic here (free of Gradio) lets the test suite
verify HTML escaping, offset-accurate highlighting and batch processing without
launching a UI.
"""
from __future__ import annotations

import html
import io
from typing import Dict, List

# Accessible, high-contrast palette (WCAG AA text on these backgrounds).
ENTITY_COLORS: Dict[str, str] = {
    "PER": "#2563eb",   # blue
    "ORG": "#7c3aed",   # violet
    "LOC": "#059669",   # green
    "MISC": "#d97706",  # amber
}
ENTITY_LABELS: Dict[str, str] = {
    "PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Misc",
}


def legend_html() -> str:
    chips = []
    for et, color in ENTITY_COLORS.items():
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin:0 10px 6px 0;font-size:13px;">'
            f'<span style="width:14px;height:14px;border-radius:4px;'
            f'background:{color};display:inline-block;"></span>'
            f'{ENTITY_LABELS[et]} ({et})</span>'
        )
    return f'<div style="margin:4px 0 10px;">{"".join(chips)}</div>'


def render_highlighted_html(text: str, entities: List[Dict]) -> str:
    """Return HTML with entity spans highlighted on the *original* text.

    Everything is HTML-escaped; entity spans are reconstructed from character
    offsets so punctuation, whitespace and repeated mentions are preserved.
    """
    if not text:
        return '<div style="color:#64748b;">No text to display.</div>'
    ents = sorted(entities, key=lambda e: e["start"])
    out, cursor = [], 0
    for e in ents:
        s, en = e["start"], e["end"]
        if s < cursor:   # skip overlaps defensively
            continue
        out.append(html.escape(text[cursor:s]))
        color = ENTITY_COLORS.get(e["type"], "#334155")
        conf = e.get("confidence")
        title = f'{ENTITY_LABELS.get(e["type"], e["type"])} ({e["type"]})'
        if conf is not None:
            title += f" · {conf:.0%}"
        out.append(
            f'<span title="{html.escape(title)}" '
            f'style="background:{color}1a;border-bottom:2px solid {color};'
            f'border-radius:3px;padding:1px 3px;color:#0f172a;">'
            f'{html.escape(text[s:en])}'
            f'<span style="font-size:9px;font-weight:700;color:{color};'
            f'vertical-align:super;margin-left:2px;">{e["type"]}</span></span>'
        )
        cursor = en
    out.append(html.escape(text[cursor:]))
    body = "".join(out).replace("\n", "<br>")
    return (f'<div style="line-height:2;font-size:15px;color:#0f172a;">{body}</div>')


def entity_table(entities: List[Dict]) -> List[List]:
    rows = []
    for e in entities:
        conf = e.get("confidence")
        rows.append([e["text"], e["type"], e["start"], e["end"],
                     f"{conf:.2%}" if conf is not None else "n/a"])
    return rows


def entity_stats(entities: List[Dict]) -> Dict:
    by_type = {et: 0 for et in ENTITY_COLORS}
    for e in entities:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    unique = len({(e["text"].lower(), e["type"]) for e in entities})
    return {
        "total_entities": len(entities),
        "unique_entities": unique,
        "by_category": by_type,
    }


def process_csv_bytes(data: bytes, text_column: str, predict_fn,
                      max_rows: int = 500, max_chars: int = 5000) -> Dict:
    """Run NER over a CSV's text column. Returns rows + summary (no persistence)."""
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data))
    if text_column not in df.columns:
        # fall back to first string-like column
        candidates = [c for c in df.columns if df[c].dtype == object]
        if not candidates:
            raise ValueError(f"No text column found. Columns: {list(df.columns)}")
        text_column = candidates[0]
    out_rows, skipped = [], 0
    for i, raw in enumerate(df[text_column].tolist()[:max_rows]):
        if not isinstance(raw, str) or not raw.strip():
            skipped += 1
            continue
        pred = predict_fn(raw[:max_chars])
        for e in pred["entities"]:
            out_rows.append({
                "row": i, "text": e["text"], "type": e["type"],
                "start": e["start"], "end": e["end"],
                "confidence": e.get("confidence"),
            })
    return {"rows": out_rows, "text_column": text_column, "skipped": skipped,
            "n_input_rows": min(len(df), max_rows)}
