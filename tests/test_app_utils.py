import io

from entitylens.app_utils import (entity_stats, entity_table, process_csv_bytes,
                                   render_highlighted_html)


def _ents():
    return [
        {"text": "Barack Obama", "type": "PER", "start": 0, "end": 12, "confidence": 0.98},
        {"text": "Hawaii", "type": "LOC", "start": 25, "end": 31, "confidence": 0.95},
    ]


def test_highlight_escapes_html():
    text = "<b>Google</b> is here"
    ents = [{"text": "Google", "type": "ORG", "start": 3, "end": 9, "confidence": 0.9}]
    out = render_highlighted_html(text, ents)
    assert "&lt;b&gt;" in out          # raw markup escaped
    assert "<b>Google</b>" not in out  # no injection


def test_highlight_preserves_offsets_and_repeats():
    text = "Paris and Paris"
    ents = [
        {"text": "Paris", "type": "LOC", "start": 0, "end": 5, "confidence": 0.9},
        {"text": "Paris", "type": "LOC", "start": 10, "end": 15, "confidence": 0.9},
    ]
    out = render_highlighted_html(text, ents)
    assert out.count("LOC") >= 2


def test_empty_text_highlight():
    assert "No text" in render_highlighted_html("", [])


def test_entity_stats():
    s = entity_stats(_ents())
    assert s["total_entities"] == 2
    assert s["by_category"]["PER"] == 1 and s["by_category"]["LOC"] == 1
    assert s["unique_entities"] == 2


def test_entity_table_shape():
    rows = entity_table(_ents())
    assert len(rows) == 2 and len(rows[0]) == 5


def test_process_csv_runs_model_and_handles_bad_rows():
    csv = "text,id\nBarack Obama in Hawaii,1\n,2\n"  # row 2 has a missing text value
    def fake_predict(t):
        return {"entities": [{"text": "X", "type": "PER", "start": 0,
                              "end": 1, "confidence": 0.9}] if t.strip() else []}
    res = process_csv_bytes(csv.encode(), "text", fake_predict)
    assert res["text_column"] == "text"
    assert res["skipped"] >= 1           # the missing-value row
    assert all("type" in r for r in res["rows"])


def test_process_csv_missing_column_falls_back():
    csv = "body\nSome text here\n"
    res = process_csv_bytes(csv.encode(), "text",
                            lambda t: {"entities": []})
    assert res["text_column"] == "body"
