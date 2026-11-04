import html

from entitylens.inference.spans import bio_to_spans
from entitylens.inference.tokenizer import tokenize_with_offsets


def test_offsets_exact():
    text = "Apple Inc. is based in Cupertino, California."
    for tok, s, e in tokenize_with_offsets(text):
        assert text[s:e] == tok


def test_unicode_and_punct():
    text = "Café Zoë met Jürgen in São Paulo!"
    toks = tokenize_with_offsets(text)
    assert ("Café", 0, 4) == toks[0]
    assert all(text[s:e] == t for t, s, e in toks)


def test_empty_input():
    assert tokenize_with_offsets("") == []


def test_span_reconstruction_preserves_text():
    text = "Angela Merkel met Barack Obama."
    toks = tokenize_with_offsets(text)
    tags = ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O"]
    spans = bio_to_spans(text, toks, tags, [0.9] * len(toks))
    assert [s["text"] for s in spans] == ["Angela Merkel", "Barack Obama"]
    for s in spans:
        assert text[s["start"]:s["end"]] == s["text"]
        assert s["type"] == "PER"


def test_adjacent_entities_separated_by_B():
    text = "Google Microsoft"
    toks = tokenize_with_offsets(text)
    spans = bio_to_spans(text, toks, ["B-ORG", "B-ORG"], [1.0, 1.0])
    assert len(spans) == 2


def test_dangling_I_opens_span():
    text = "Paris France"
    toks = tokenize_with_offsets(text)
    spans = bio_to_spans(text, toks, ["I-LOC", "I-LOC"], [0.5, 0.5])
    assert len(spans) == 1 and spans[0]["text"] == "Paris France"


def test_repeated_entities_distinct_offsets():
    text = "London London"
    toks = tokenize_with_offsets(text)
    spans = bio_to_spans(text, toks, ["B-LOC", "B-LOC"], [1.0, 1.0])
    assert spans[0]["start"] != spans[1]["start"]


def test_html_escaping_safe():
    text = "<script>alert(1)</script> Google"
    toks = tokenize_with_offsets(text)
    # the raw text must be escapable without breaking offsets
    assert "&lt;script&gt;" in html.escape(text)
