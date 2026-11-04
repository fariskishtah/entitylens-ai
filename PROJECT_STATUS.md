# EntityLens AI — Project Status

_Intelligent Named Entity Recognition & Text Analysis Platform._
Independent project; does not touch FinTriage AI, OralVision AI, or the RAG platform.

**Status: COMPLETE** (public release live). Legend: ✅ done · ⚠️ caveat

## Public links

- **GitHub (PUBLIC):** https://github.com/fariskishtah/entitylens-ai
- **HF Model (PUBLIC):** https://huggingface.co/fariskishtah/entitylens-ai
- **HF Space / live demo (PUBLIC):** https://huggingface.co/spaces/fariskishtah/entitylens-ai

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| 1 | Dataset verification + audit | ✅ |
| 2 | Project scaffold + env | ✅ |
| 3 | Preprocessing, BIO, GloVe, char-CNN OOV | ✅ |
| 4 | Four architectures + 32 tests | ✅ |
| 5 | Full training (3 RNN local MPS + BERT Kaggle T4) | ✅ |
| 6 | Evaluation + validation-based selection | ✅ |
| 7 | Gradio app (+ static transformers.js app) | ✅ |
| 8 | QA + real-model inference tests | ✅ |
| 9 | Public deployment (model repo + static Space) | ✅ |
| 10 | GitHub + README + screenshots + audit | ✅ |

## Final results (CoNLL-2003 test, strict seqeval)

| Model | Val F1 | Test F1 | PER | ORG | LOC | MISC | invalid BIO | params |
|---|---|---|---|---|---|---|---|---|
| LSTM | 0.8513 | 0.7772 | 0.830 | 0.690 | 0.837 | 0.695 | 0.82% | 2.37M |
| BiLSTM | 0.9030 | 0.8439 | 0.878 | 0.799 | 0.895 | 0.752 | 0.27% | 2.64M |
| BiLSTM+CRF | 0.9206 | 0.8656 | 0.913 | 0.820 | 0.906 | 0.767 | 0.00% | 2.64M |
| **BERT ⭐** | **0.9537** | **0.9178** | 0.963 | 0.902 | 0.931 | 0.819 | 0.075% | 107.7M |

Production model: **BERT (bert-base-cased)**, selected on validation F1.
Tests: **32/32 passing**. CRF eliminates invalid BIO transitions entirely.

## Deployment notes / caveats

- HF now requires a **PRO (paid)** subscription for free-CPU **Gradio** Spaces
  (account is not PRO). Per instructions, no subscription was purchased; instead
  the model ships as a **free static Space** running the quantized ONNX model
  in-browser via transformers.js. The full Gradio `app.py` runs locally.
- ⚠️ Live in-browser inference and model load were verified in-session (model
  reached "ready", first inference ran). After a display-aggregation fix (deployed),
  the fix's logic was re-verified by local simulation and by local ONNX inference,
  but the post-fix highlight could not be re-screenshotted because the browser pane
  rate-limited repeated ~100 MB model downloads. A fresh visitor downloads once and
  sees full highlighting.
- Corpus tokens are **not** redistributed on GitHub (Reuters licensing);
  regenerate the frozen splits with `python scripts/prepare_data.py`.
