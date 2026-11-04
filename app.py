"""EntityLens AI — Intelligent Named Entity Recognition & Text Analysis.

A professional Gradio interface over the selected production NER model:
entity highlighting with exact character offsets, an extracted-entities table,
live statistics, an entity inspector, batch CSV processing, honest model-
performance metrics and result export.

Model directory resolution order:
  1. $ENTITYLENS_MODEL_DIR
  2. artifacts/production
  3. first available of artifacts/{bilstm_crf,bilstm,transformer/model,lstm}
If no trained model is found the app still launches and explains how to train
one (no placeholder predictions are ever shown).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from entitylens.app_utils import (ENTITY_COLORS, entity_stats, entity_table,  # noqa: E402
                                  legend_html, process_csv_bytes,
                                  render_highlighted_html)
from entitylens.labels import ENTITY_DESCRIPTIONS  # noqa: E402

NAVY = "#0f172a"
MAX_CHARS = 10000

EXAMPLES = [
    "Barack Obama was born in Hawaii and later worked in Chicago before becoming President of the United States.",
    "Apple and Microsoft are competing with Google and Amazon in the artificial intelligence market.",
    "The European Union and the United Nations met in Geneva to discuss the conflict in the Middle East.",
    "Lionel Messi left FC Barcelona to join Paris Saint-Germain, surprising fans across Europe.",
    "Dr. Angela Merkel, the former German Chancellor, visited the White House in Washington last year.",
    "Tesla, led by Elon Musk, announced a new factory in Berlin while Toyota expanded in Texas.",
]


_SELECTED_MAP = {"LSTM": "lstm", "BiLSTM": "bilstm", "BiLSTM+CRF": "bilstm_crf",
                 "Transformer": "transformer/model"}


def _resolve_model_dir() -> Path | None:
    env = os.environ.get("ENTITYLENS_MODEL_DIR")
    if env and Path(env).exists():
        return Path(env)
    # prefer an explicit production dir, else the model chosen by evaluation
    prod = ROOT / "artifacts" / "production"
    if prod.exists() and ((prod / "best.pt").exists() or (prod / "config.json").exists()):
        return prod
    comp = ROOT / "results" / "model_comparison.json"
    if comp.exists():
        try:
            sel = json.loads(comp.read_text())["selected_model"]
            cand = ROOT / "artifacts" / _SELECTED_MAP.get(sel, "")
            if cand.exists() and ((cand / "best.pt").exists() or (cand / "config.json").exists()):
                return cand
        except Exception:
            pass
    for name in ["transformer/model", "bilstm_crf", "bilstm", "lstm"]:
        c = ROOT / "artifacts" / name
        if c.exists() and ((c / "best.pt").exists() or (c / "config.json").exists()):
            return c
    return None


class _App:
    def __init__(self):
        self.predictor = None
        self.model_dir = _resolve_model_dir()
        self.meta = {}
        if self.model_dir is not None:
            try:
                from entitylens.inference.predictor import Predictor
                self.predictor = Predictor.load(self.model_dir)
                self.meta = self._load_meta()
            except Exception as e:  # pragma: no cover - startup robustness
                print(f"[EntityLens] model load failed: {e}")

    def _load_meta(self) -> dict:
        comp = ROOT / "results" / "model_comparison.json"
        if comp.exists():
            return json.loads(comp.read_text())
        return {}

    def ready(self) -> bool:
        return self.predictor is not None

    def analyze(self, text: str):
        if not self.ready():
            warn = ("<div style='color:#b91c1c;'>No trained model is loaded. "
                    "Train one (see README) or set $ENTITYLENS_MODEL_DIR.</div>")
            return warn, [], "{}", gr.update(choices=[], value=None)
        text = (text or "").strip()
        if not text:
            return ("<div style='color:#64748b;'>Enter some English text to begin.</div>",
                    [], "{}", gr.update(choices=[], value=None))
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        pred = self.predictor.predict(text)
        ents = pred["entities"]
        html_out = legend_html() + render_highlighted_html(text, ents)
        table = entity_table(ents)
        stats = entity_stats(ents)
        choices = [f'{i}: {e["text"]} ({e["type"]})' for i, e in enumerate(ents)]
        self._last = {"text": text, "entities": ents}
        return html_out, table, json.dumps(stats, indent=2), gr.update(choices=choices, value=None)

    def inspect(self, choice: str):
        if not choice or not getattr(self, "_last", None):
            return "Select an entity from the table above."
        idx = int(choice.split(":", 1)[0])
        e = self._last["entities"][idx]
        conf = e.get("confidence")
        conf_txt = f"{conf:.2%} (raw model score, not calibrated)" if conf is not None else "n/a"
        return (f"**Span:** `{e['text']}`\n\n"
                f"**Category:** {e['type']}\n\n"
                f"**Character offsets:** {e['start']}–{e['end']}\n\n"
                f"**Confidence:** {conf_txt}\n\n"
                f"**What this label means:** {ENTITY_DESCRIPTIONS.get(e['type'], '')}")

    def export_json(self):
        if not getattr(self, "_last", None):
            return None
        tmp = Path(tempfile.gettempdir()) / "entitylens_results.json"
        tmp.write_text(json.dumps(self._last, indent=2))
        return str(tmp)

    def batch(self, file, column):
        if not self.ready():
            return [], None, "No trained model loaded."
        if file is None:
            return [], None, "Upload a CSV first."
        data = Path(file).read_bytes()
        try:
            res = process_csv_bytes(data, column or "text", self.predictor.predict)
        except Exception as e:
            return [], None, f"Error: {e}"
        rows = [[r["row"], r["text"], r["type"], r["start"], r["end"],
                 f'{r["confidence"]:.2%}' if r["confidence"] is not None else "n/a"]
                for r in res["rows"]]
        tmp = Path(tempfile.gettempdir()) / "entitylens_batch.json"
        tmp.write_text(json.dumps(res, indent=2))
        msg = (f'Processed {res["n_input_rows"]} rows from column '
               f'"{res["text_column"]}" · {len(res["rows"])} entities · '
               f'{res["skipped"]} rows skipped.')
        return rows, str(tmp), msg

    def performance_md(self) -> str:
        if not self.meta:
            return ("_Model comparison metrics will appear here once "
                    "`scripts/evaluate_all.py` has run._")
        rows = self.meta.get("comparison", [])
        sel = self.meta.get("selected_model", "?")
        head = ("| Model | Precision | Recall | Entity F1 | Train (s) | "
                "Latency (ms) | Params | Size (MB) |\n"
                "|---|---|---|---|---|---|---|---|\n")
        body = ""
        for r in rows:
            body += (f'| {r.get("model")} | {r.get("precision",0):.4f} | '
                     f'{r.get("recall",0):.4f} | **{r.get("f1",0):.4f}** | '
                     f'{r.get("train_time_sec","–")} | {r.get("latency_ms","–")} | '
                     f'{r.get("params","–")} | {r.get("size_mb","–")} |\n')
        return (f"### Four-model comparison (test set, strict seqeval)\n\n"
                f"{head}{body}\n**Production model:** `{sel}` — "
                f"{self.meta.get('selection_reason','selected on validation F1')}")


def build_ui() -> gr.Blocks:
    app = _App()
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(theme=theme, title="EntityLens AI", css="""
        .gradio-container {max-width: 1100px !important;}
        #hero {background: linear-gradient(135deg,#0f172a,#1e3a8a);
               color:white;padding:22px 26px;border-radius:14px;margin-bottom:8px;}
        #hero h1 {margin:0;font-size:26px;} #hero p {margin:6px 0 0;opacity:.85;}
    """) as demo:
        gr.HTML(f"""<div id='hero'><h1>🔎 EntityLens AI</h1>
            <p>Intelligent Named Entity Recognition &amp; Text Analysis</p></div>""")
        if not app.ready():
            gr.Markdown("> ⚠️ **No trained model loaded.** The interface is live, "
                        "but predictions need a trained checkpoint. See the README "
                        "to train models, then restart.")
        else:
            gr.Markdown(f"<sub>Production model: <b>{app.model_dir.name}</b> · "
                        f"backend: <b>{app.predictor.backend}</b></sub>")

        with gr.Tab("Analyze"):
            inp = gr.Textbox(label="English text", lines=5,
                             placeholder="Paste text with people, organizations, locations…")
            with gr.Row():
                btn = gr.Button("Analyze", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(EXAMPLES, inputs=inp, label="Example texts")
            out_html = gr.HTML(label="Highlighted entities")
            with gr.Row():
                with gr.Column(scale=2):
                    table = gr.Dataframe(
                        headers=["Entity", "Type", "Start", "End", "Confidence"],
                        label="Extracted entities", interactive=False, wrap=True)
                with gr.Column(scale=1):
                    stats = gr.Code(label="Statistics", language="json")
            with gr.Row():
                picker = gr.Dropdown(label="Entity inspector — pick a span", choices=[])
                export_btn = gr.Button("⬇ Export results (JSON)")
            inspector = gr.Markdown()
            export_file = gr.File(label="Download", visible=True)

            btn.click(app.analyze, inp, [out_html, table, stats, picker])
            inp.submit(app.analyze, inp, [out_html, table, stats, picker])
            clear.click(lambda: ("", [], "{}", gr.update(choices=[])), None,
                        [out_html, table, stats, picker])
            picker.change(app.inspect, picker, inspector)
            export_btn.click(app.export_json, None, export_file)

        with gr.Tab("Batch (CSV)"):
            gr.Markdown("Upload a CSV with a text column; download extracted "
                        "entities as JSON. Files are processed in-memory and not stored.")
            csv_in = gr.File(label="CSV file", file_types=[".csv"])
            col = gr.Textbox(label="Text column name", value="text")
            run_batch = gr.Button("Process CSV", variant="primary")
            batch_msg = gr.Markdown()
            batch_table = gr.Dataframe(
                headers=["Row", "Entity", "Type", "Start", "End", "Confidence"],
                label="Results", interactive=False, wrap=True)
            batch_file = gr.File(label="Download JSON")
            run_batch.click(app.batch, [csv_in, col], [batch_table, batch_file, batch_msg])

        with gr.Tab("Model performance"):
            gr.Markdown(app.performance_md())

        with gr.Tab("About"):
            gr.Markdown(
                "EntityLens AI recognizes four entity types — **Person (PER)**, "
                "**Organization (ORG)**, **Location (LOC)** and **Miscellaneous "
                "(MISC)** — using models trained on the CoNLL-2003 benchmark. "
                "Highlighting uses exact character offsets, so the original text "
                "(punctuation, spacing, repeated mentions) is always preserved. "
                "Confidence values are raw model scores and are **not** calibrated "
                "probabilities.")
    return demo


if __name__ == "__main__":
    import os as _os
    build_ui().launch(
        server_name=_os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(_os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=_os.environ.get("GRADIO_SHARE") == "1",
        show_api=False,
    )
