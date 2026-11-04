from entitylens.data.alignment import tokenize_and_align_labels
from entitylens.evaluation.metrics import bio_transition_validity, entity_level_report
from entitylens.labels import PAD_LABEL_ID


def test_entity_level_metrics_perfect():
    y = [["B-PER", "I-PER", "O", "B-LOC"]]
    rep = entity_level_report(y, y)
    assert rep["micro"]["f1"] == 1.0
    assert rep["per_type"]["PER"]["f1"] == 1.0


def test_entity_level_metrics_boundary_error():
    true = [["B-PER", "I-PER", "O"]]
    pred = [["B-PER", "O", "O"]]  # wrong boundary -> span mismatch
    rep = entity_level_report(true, pred)
    assert rep["micro"]["f1"] < 1.0  # strict: partial overlap is not a match


def test_bio_validity_flags_illegal():
    pred = [["O", "I-PER", "O"]]  # illegal dangling I-
    v = bio_transition_validity(pred)
    assert v["invalid_transitions"] == 1


def test_subword_alignment_first_subword_only():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    batch = {"tokens": [["Washington", "unbelievably", "here"]],
             "ner_tags": [[5, 0, 0]]}  # B-LOC, O, O
    out = tokenize_and_align_labels(batch, tok, max_length=32)
    labels = out["labels"][0]
    word_ids = out.word_ids(0)
    # special tokens and continuation subwords are ignored (-100)
    for pos, wid in enumerate(word_ids):
        if wid is None:
            assert labels[pos] == PAD_LABEL_ID
    # exactly 3 non-ignored labels (one per word, first subword)
    assert sum(1 for l in labels if l != PAD_LABEL_ID) == 3
    assert labels[1] == 5  # first subword of "Washington" keeps B-LOC
