import torch

from entitylens.labels import LABEL2ID, NUM_LABELS, PAD_LABEL_ID
from entitylens.models import BiLSTMCRFTagger, BiLSTMTagger, LSTMTagger

B, T, W, NW, NC = 3, 8, 10, 40, 25


def _batch():
    torch.manual_seed(0)
    word = torch.randint(0, NW, (B, T))
    char = torch.randint(0, NC, (B, T, W))
    lengths = torch.tensor([8, 5, 2])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    labels = torch.randint(0, NUM_LABELS, (B, T))
    labels[~mask] = PAD_LABEL_ID
    return word, char, lengths, mask, labels


def _model(cls):
    return cls(NW, NC, hidden_dim=16, word_emb_dim=8, use_char=True)


def test_logits_dims_and_loss_backprop():
    word, char, lengths, mask, labels = _batch()
    for cls in (LSTMTagger, BiLSTMTagger):
        m = _model(cls)
        out = m(word, char, lengths, mask=mask, labels=labels)
        assert out["logits"].shape == (B, T, NUM_LABELS)
        out["loss"].backward()
        assert out["loss"].item() > 0


def test_decode_lengths_match():
    word, char, lengths, mask, _ = _batch()
    for cls in (LSTMTagger, BiLSTMTagger, BiLSTMCRFTagger):
        m = _model(cls)
        dec = m.decode(word, char, lengths, mask)
        assert [len(d) for d in dec] == lengths.tolist()


def test_crf_loss_and_constraints():
    word, char, lengths, mask, labels = _batch()
    m = BiLSTMCRFTagger(NW, NC, hidden_dim=16, word_emb_dim=8)
    out = m(word, char, lengths, mask=mask, labels=labels)
    out["loss"].backward()
    m.apply_constraints()
    # illegal start into I- must be heavily penalized
    assert m.crf.start_transitions.data[LABEL2ID["I-PER"]].item() <= -9999
    # illegal transition O -> I-PER
    assert m.crf.transitions.data[LABEL2ID["O"], LABEL2ID["I-PER"]].item() <= -9999


def test_checkpoint_roundtrip(tmp_path):
    word, char, lengths, mask, _ = _batch()
    m = _model(BiLSTMTagger)
    m.eval()
    before = m.decode(word, char, lengths, mask)
    path = tmp_path / "m.pt"
    torch.save({"model_state": m.state_dict(), "arch": m.arch}, path)
    m2 = _model(BiLSTMTagger)
    m2.load_state_dict(torch.load(path, weights_only=False)["model_state"])
    m2.eval()
    after = m2.decode(word, char, lengths, mask)
    assert before == after
