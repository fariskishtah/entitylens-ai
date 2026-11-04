from entitylens.labels import (ID2LABEL, LABEL2ID, LABELS, NUM_LABELS,
                                is_valid_transition, split_tag)


def test_label_mapping_roundtrip():
    assert NUM_LABELS == 9
    for i, lab in enumerate(LABELS):
        assert LABEL2ID[lab] == i
        assert ID2LABEL[i] == lab


def test_split_tag():
    assert split_tag("B-PER") == ("B", "PER")
    assert split_tag("I-LOC") == ("I", "LOC")
    assert split_tag("O") == ("O", "")


def test_valid_transitions():
    assert is_valid_transition("O", "B-PER")
    assert is_valid_transition("B-PER", "I-PER")
    assert is_valid_transition("I-PER", "I-PER")
    assert is_valid_transition("B-PER", "O")


def test_invalid_transitions():
    # I- cannot start a span or follow a mismatched type / O
    assert not is_valid_transition("O", "I-PER")
    assert not is_valid_transition("B-ORG", "I-PER")
    assert not is_valid_transition("I-LOC", "I-ORG")
