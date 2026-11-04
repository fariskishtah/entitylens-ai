import torch

from entitylens.data.rnn_dataset import RNNNerDataset, make_collate
from entitylens.data.vocab import PAD, UNK, build_vocabularies
from entitylens.labels import NUM_LABELS, PAD_LABEL_ID

# Synthetic fixtures (not from CoNLL-2003) covering B-ORG, B-MISC, O and a
# multi-token B-PER/I-PER span.
EXAMPLES = [
    {"tokens": ["Acme", "hired", "Canadian", "staff"], "ner_tags": [3, 0, 7, 0]},
    {"tokens": ["Jane", "Doe"], "ner_tags": [1, 2]},
]


def test_vocab_built_from_train_only_and_specials_first():
    v = build_vocabularies(EXAMPLES)
    assert v.word.itos[0] == PAD and v.word.itos[1] == UNK
    assert v.char.itos[0] == PAD and v.char.itos[1] == UNK


def test_oov_maps_to_unk():
    v = build_vocabularies(EXAMPLES)
    assert v.encode_word("totally-unseen-word") == v.word.unk_id


def test_char_encoding_covers_oov_words():
    v = build_vocabularies(EXAMPLES)
    ids = v.encode_chars("German")
    assert len(ids) == len("German")


def test_collate_shapes_and_mask():
    v = build_vocabularies(EXAMPLES)
    ds = RNNNerDataset(EXAMPLES, v)
    collate = make_collate(v.word.pad_id, v.char.pad_id, v.max_word_len)
    batch = collate([ds[0], ds[1]])
    B, T = 2, 4
    assert batch["word_ids"].shape == (B, T)
    assert batch["char_ids"].shape == (B, T, v.max_word_len)
    assert batch["labels"].shape == (B, T)
    assert batch["mask"].dtype == torch.bool
    # padded positions: mask False and label == ignore index
    assert batch["mask"][1, 2:].sum() == 0
    assert (batch["labels"][1, 2:] == PAD_LABEL_ID).all()
    assert batch["lengths"].tolist() == [4, 2]


def test_vocab_save_load_roundtrip(tmp_path):
    v = build_vocabularies(EXAMPLES)
    p = tmp_path / "vocab.json"
    v.save(p)
    from entitylens.data.vocab import Vocabularies
    v2 = Vocabularies.load(p)
    assert v2.word.itos == v.word.itos
    assert v2.encode_word("Acme") == v.encode_word("Acme")
