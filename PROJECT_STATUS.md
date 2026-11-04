# EntityLens AI — Project Status

_Intelligent Named Entity Recognition & Text Analysis Platform._
Independent project; does not touch FinTriage AI, OralVision AI, or the RAG platform.

Legend: ✅ completed · 🔄 in progress · ⛔ blocked / needs decision

## Milestones

| # | Milestone | State | Evidence |
|---|-----------|-------|----------|
| 1 | Dataset verification + audit | ✅ | `data/processed/data_audit.json`, `dataset_provenance.json` |
| 2 | Project scaffold + env | ✅ | `src/entitylens/`, `.venv` (py3.12, torch 2.5.1 MPS) |
| 3 | Preprocessing, BIO, alignment, GloVe, OOV | ✅ | `data/`, `models/components.py` (char-CNN), 87.6% GloVe coverage |
| 4 | Four architectures + tests + smoke | ✅ | `models/`, `tests/` (32 passing), all 4 train paths smoke-tested |
| 5 | Full training (3 RNNs local, BERT) | 🔄 | RNNs training on MPS; BERT path pending decision |
| 6 | Evaluation + selection | 🔄 | `scripts/evaluate_all.py` ready; runs after training |
| 7 | Gradio application | ✅ | `app.py` + `src/entitylens/app_utils.py` (tested) |
| 8 | Integration + visual QA | 🔄 | pending trained production model |
| 9 | Private HF deployment | ⛔ | needs trained model + user approval (visibility/cost) |
| 10 | GitHub + README + audit | 🔄 | repo files written; README metrics pending training |

## Verified dataset facts (CoNLL-2003, `tomaarsen/conll2003` parquet mirror)

- Splits (official, frozen): train **14,041** / val **3,250** / test **3,453** sentences.
- Tokens: 203,621 / 51,362 / 46,435. Entities: 23,499 / 5,942 / 5,648.
- Tagging: **IOB2** confirmed — 0 invalid transitions in any split.
- Labels: `O, B-/I-{PER,ORG,LOC,MISC}` (9 classes).
- Cross-split OOV (token): val 7.54%, test 10.87% → handled by char-CNN + GloVe + (BERT) subwords.
- Known benchmark quirk (documented, not "fixed"): small cross-split sentence
  overlap (train↔val 129, train↔test 78) — mostly repeated short sport-score
  lines. Official splits are preserved per the evaluation protocol.

## Engineering decisions

- One canonical `label2id`/`id2label` shared by all four models + the app.
- Vocab + embeddings built from **train only** (no leakage).
- CRF: real `pytorch-crf` with Viterbi decoding + hard IOB2 transition
  constraints re-applied after every optimizer step.
- Transformer: first-subword labels only; metrics realigned to word level.
- Model selection by **validation** F1; test set evaluated once at the end.
- Credentials present on this machine: Kaggle, HF (cached), GitHub (`fariskishtah`).

## Live training progress (resume point)

- **Public release AUTHORIZED** by user (GitHub public + HF public Space + public model). CPU free hosting preferred; no paid hardware.
- **LSTM**: training locally on MPS, reached **val F1 ≈ 0.851** by epoch 12, converging (early stop pending). ~2 min/epoch.
- **BiLSTM, BiLSTM+CRF**: queued after LSTM (sequential via `scripts/train_all_rnn.py`, PID-tracked).
- **BERT (Kaggle GPU)**: all 4 epochs completed (checkpoints 439→1756); kernel finalizing eval/save. Pull with `python scripts/kaggle_launch.py --pull`, place under `artifacts/transformer/model/`.
- A background waiter blocks until local training ends AND the Kaggle kernel leaves RUNNING, then resumes the pipeline.

## Verified training results (final)

| Model | val F1 | test F1 | test P | test R | invalid BIO | params | latency |
|---|---|---|---|---|---|---|---|
| LSTM | 0.8513 | 0.7772 | 0.786 | 0.769 | 0.82% | 2.37M | 2.0ms |
| BiLSTM | 0.9030 | 0.8439 | 0.846 | 0.842 | 0.27% | 2.64M | 2.9ms |
| BiLSTM+CRF | **0.9206** | **0.8656** | 0.874 | 0.857 | **0.00%** | 2.64M | 3.2ms |
| BERT | _(pending pull)_ | | | | | 108M | |

Per-class test F1 (CRF): PER 0.9125 · ORG 0.8201 · LOC 0.9056 · MISC 0.767.
Production candidate by validation F1: **BiLSTM+CRF** (CPU-friendly, 3ms/sentence) — confirm after BERT val F1 known.

## Resume point (exact pending operations)

1. **BERT pull IN PROGRESS** → background task downloading only `artifacts/transformer/model/*`
   from Kaggle kernel `fariskishtah/entitylens-transformer-training` into `/tmp/bertpull/...`.
   Needed files: model.safetensors, config.json (has id2label/label2id), tokenizer.json,
   tokenizer_config.json, special_tokens_map.json, vocab.txt. On completion → move to
   `artifacts/transformer/model/`, verify `Predictor.load` works, then re-run
   `scripts/evaluate_all.py` (adds BERT row, re-selects by val F1).
2. figures (`scripts/make_figures.py`) → fill README `<!-- RESULTS_TABLE -->`,
   `<!-- PER_CLASS_TABLE -->`, `<!-- SCREENSHOTS -->`.
3. launch app, capture screenshots (desktop + mobile), save to `assets/`.
4. security audit (secrets/tokens/raw data) → `git init` → **PUBLIC** GitHub `fariskishtah/entitylens-ai`.
5. `scripts/deploy_hf.py --push --public` → **PUBLIC** HF Space → live verify unauthenticated.

Public release AUTHORIZED. Free CPU hosting only. Do not retrain. Do not touch other projects.
