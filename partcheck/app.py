"""PartCheck: web app (Gradio wiring only; all logic and HTML live in triage_ui.py so they can be tested without Gradio).

Tab 1  Check an order       : a customer's review and/or one picture -> issue? random picture? does the picture match the review? -> recommendation
Tab 2  Sentiment dashboard  : many reviews -> how customers feel, what drives the unhappy ones, which reviews to look at

Run:   python app.py          (needs the `artifacts/` folder produced by the notebook)
"""
import os
import json
import gradio as gr

import triage_ui as ui

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("ARTIFACTS", os.path.join(ROOT, "artifacts"))
EX_DIR = os.path.join(ROOT, "examples_cache")
MODEL_REPO = os.environ.get("MODEL_REPO", "unnati002/review-triage-files")   # public Hugging Face repo that holds the trained models (too large for GitHub)


def ensure_artifacts():
    """First run on a new machine or server: download the trained models from the Hugging Face repo (about 270 MB, once)."""
    global ART
    if os.path.exists(os.path.join(ART, "issue_config.json")):
        return
    from huggingface_hub import snapshot_download
    print("Downloading the trained models from", MODEL_REPO, "(first run only)...")
    local = snapshot_download(MODEL_REPO, local_dir=os.path.join(ROOT, "_models"), allow_patterns=["artifacts/*", "artifacts/**", "demo_auto_reviews.csv", "examples.json"])
    ART = os.path.join(local, "artifacts")
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False
MAX_ROWS = int(os.environ.get("MAX_ROWS", 500 if HAS_GPU else 80))

_cache = {}


def models():
    if "m" not in _cache:
        ensure_artifacts()
        _cache["m"] = ui.Models.load(ART)
    return _cache["m"]


def find_file(name):
    for base in (ROOT, ART):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- example pictures (fetched at start-up, not redistributed)
def example_pictures():
    """Catalogue product photos listed in examples.json are downloaded once; a synthetic 'random' screenshot is drawn locally."""
    os.makedirs(EX_DIR, exist_ok=True)
    out = {}
    try:
        items = json.load(open(os.path.join(ROOT, "examples.json"), encoding="utf-8"))
    except Exception:
        items = []
    try:
        import requests
        from PIL import Image
        import io
        for it in items:
            path = os.path.join(EX_DIR, it["name"] + ".jpg")
            if not os.path.exists(path):
                try:
                    r = requests.get(it["url"], timeout=10)
                    im = Image.open(io.BytesIO(r.content)).convert("RGB")
                    im.thumbnail((640, 640))
                    im.save(path, quality=88)
                except Exception:
                    continue
            out[it["name"]] = path
    except Exception:
        pass
    shot = os.path.join(EX_DIR, "random_screenshot.png")
    if not os.path.exists(shot):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (480, 360), (245, 245, 247))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 480, 40], fill=(60, 120, 220))
        for i, t in enumerate(["Messages", "Hey, are we still on for lunch?", "Sure, 12:30 at the usual place", "Great, see you there!", "Running 5 min late", "No problem"]):
            d.text((20, 60 + i * 40), t, fill=(30, 30, 30))
        im.save(shot)
    out["random_screenshot"] = shot
    return out


# ---------------------------------------------------------------- theme (warm white, navy text, one orange accent)
def make_theme():
    orange = gr.themes.Color(c50="#FFF7E6", c100="#FFEBC2", c200="#FFD98A", c300="#FFC24D", c400="#FFAD1F", c500="#FF9900",
                             c600="#E88B00", c700="#C47400", c800="#9B5C00", c900="#7A4A00", c950="#4D2E00")
    try:
        t = gr.themes.Base(primary_hue=orange, secondary_hue="slate", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
        light = dict(body_background_fill="#FBF8F3", body_text_color="#0F1A2B", body_text_color_subdued="#4A5868",
                     block_background_fill="#FFFFFF", block_border_color="#DAD4CA", block_label_text_color="#4A5868", block_title_text_color="#0F1A2B",
                     input_background_fill="#FFFFFF", input_border_color="#B8B1A6", input_placeholder_color="#6B7785",
                     button_primary_background_fill="#FF9900", button_primary_background_fill_hover="#E88B00",
                     button_primary_text_color="#0F1A2B", button_primary_border_color="#E88B00",
                     button_secondary_background_fill="#FFFFFF", button_secondary_text_color="#0F1A2B", button_secondary_border_color="#B8B1A6",
                     checkbox_label_background_fill="#FFFFFF", checkbox_label_text_color="#0F1A2B",
                     checkbox_label_background_fill_selected="#232F3E", checkbox_label_text_color_selected="#FFFFFF",
                     checkbox_background_color_selected="#FF9900", checkbox_border_color_selected="#FF9900",
                     table_even_background_fill="#FFFFFF", table_odd_background_fill="#FBF8F3", table_row_focus="#FFF4E0", color_accent_soft="#FFF4E0")
        for k, v in list(light.items()) + [(k + "_dark", v) for k, v in light.items()]:
            if hasattr(t, k):
                setattr(t, k, v)
        return t
    except Exception:
        return gr.themes.Default(primary_hue="orange", neutral_hue="slate")


FORCE_LIGHT = "() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }"

HERO = "<div class='pc-hero'><h1>PartCheck</h1><span class='sub'>Automotive customer feedback</span></div>"

# ---------------------------------------------------------------- actions
def check_one(text, image, ordered=""):
    yield "<div class='pc-empty'>Checking the order...</div>"
    try:
        yield ui.render_order(ui.assess_order(text, image, models(), ordered))
    except Exception as e:
        yield ui.banner("err", f"Something went wrong while checking this order ({type(e).__name__}). Try again, or use a different picture.")


NO_PLOT = gr.update(value=None, visible=False)


def run_reviews(pasted, sheet, use_demo=False):
    yield (None, ui.banner("info", "Analysing the reviews: the first run takes a minute while the models load..."), "", NO_PLOT, NO_PLOT, NO_PLOT, NO_PLOT, "", None)
    try:
        demo_path = find_file("demo_auto_reviews.csv") if use_demo else None
        if use_demo and demo_path is None:
            raise ValueError("The demo file demo_auto_reviews.csv was not found next to app.py.")
        df, note = ui.read_table(pasted, getattr(sheet, "name", sheet), demo_path)
        res = ui.analyse_reviews(df, models(), fast=not HAS_GPU, max_rows=MAX_ROWS)
    except Exception as e:
        yield (None, ui.banner("err", f"Could not analyse this input. {e}"), "", NO_PLOT, NO_PLOT, NO_PLOT, NO_PLOT, "", None)
        return
    q = res["q"]
    show = lambda fig: gr.update(value=fig, visible=fig is not None)
    notes = " ".join(([note] if note else []) + res["notes"])
    yield (res, ui.banner("warn" if res["notes"] else "ok", f"Done: {len(q)} reviews analysed. {notes}".strip()), ui.render_kpis(q), show(ui.fig_split(q)), show(ui.fig_trend(q)),
           show(ui.fig_products(q)), show(ui.fig_drivers(q)), ui.render_queue(q, "Needs review"), ui.export_csv(q))


def run_demo():
    yield from run_reviews("", None, use_demo=True)


def refilter(res, show):
    return "" if res is None else ui.render_queue(res["q"], show)


# ---------------------------------------------------------------- app
def build():
    major = int(gr.__version__.split(".")[0])
    bk, lk = {"title": "PartCheck"}, {}
    theme = make_theme()
    if major >= 6:
        lk = {"theme": theme, "css": ui.CSS, "js": FORCE_LIGHT}
    else:
        bk.update(theme=theme, css=ui.CSS, js=FORCE_LIGHT)
    pics = example_pictures()
    ex = [["It doesn't fit my 2018 F-150. The bolt holes are off and the listing said it would fit.", None, ""],
          ["Looks great and feels well made, but it doesn't fit my 2015 Civic.", None, ""],
          ["The bottle of motor oil arrived leaking and the box is soaked.", pics.get("tyre"), ""],
          ["This is not what I expected, it does not look right.", pics.get("tyre"), "Valvoline synthetic motor oil 5 quart"],
          ["These floor mats are great, perfect fit and easy to clean.", pics.get("floor_mat"), ""],
          ["Great product, thank you!", pics.get("random_screenshot"), ""]]
    ex = [e for e in ex if e[1] is None or os.path.exists(e[1])]
    with gr.Blocks(**bk) as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            # ================= tab 1: one order =================
            with gr.Tab("Check an order"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML("<div class='pc-h2'>Customer review and picture</div><p class='pc-sub'>Add a review, a picture, or both. The product ordered is optional.</p>")
                        t = gr.Textbox(lines=4, max_lines=6, label="Customer review", placeholder="Paste or type the review")
                        im = gr.Image(type="pil", label="Customer picture (optional)", height=210)
                        od = gr.Textbox(lines=1, label="Product ordered (optional)", placeholder="For example: Bosch wiper blades 24 inch")
                        b = gr.Button("Analyse order", variant="primary", size="lg")
                        gr.Examples(examples=ex, inputs=[t, im, od], label="Try an example", examples_per_page=6)
                    with gr.Column(scale=1):
                        gr.HTML("<div class='pc-h2'>Order assessment</div><p class='pc-sub'>The review and the picture are checked separately, then combined into one priority for the returns team.</p>")
                        card = gr.HTML("<div class='pc-empty'>The assessment appears here.</div>")
                b.click(check_one, [t, im, od], card)
            # ================= tab 2: many reviews =================
            with gr.Tab("Sentiment dashboard"):
                gr.HTML("<div class='pc-h2'>How do customers feel?</div><p class='pc-sub'>Upload a spreadsheet of reviews (one column with the review text is enough; rating, date and product make the charts richer), or try the demo.</p>")
                with gr.Row():
                    sheet = gr.File(label="Reviews (Excel or CSV)", file_types=[".csv", ".xlsx", ".xls"], type="filepath")
                    pasted = gr.Textbox(lines=5, label="...or paste reviews, one per line")
                with gr.Row():
                    go = gr.Button("Analyse reviews", variant="primary")
                    demo_btn = gr.Button("Demo: 298 real reviews")
                status = gr.HTML()
                kpi = gr.HTML()
                with gr.Row():
                    p1 = gr.Plot(visible=False, show_label=False)
                    p2 = gr.Plot(visible=False, show_label=False)
                with gr.Row():
                    p3 = gr.Plot(visible=False, show_label=False)
                p4 = gr.Plot(visible=False, show_label=False)
                gr.HTML("<div class='pc-h2' style='margin-top:10px'>Reviews to look at</div>")
                show = gr.Radio(["Needs review", "Everything"], value="Needs review", label="Show")
                queue = gr.HTML()
                csv = gr.File(label="Download all reviews with sentiment, issue, part and priority (CSV)", interactive=False)
                res_state = gr.State(None)
                outs = [res_state, status, kpi, p1, p2, p3, p4, queue, csv]
                go.click(run_reviews, [pasted, sheet], outs)
                demo_btn.click(run_demo, None, outs)
                show.change(refilter, [res_state, show], queue)
    return demo, lk


if __name__ == "__main__":
    app, kw = build()
    app.launch(**kw)
