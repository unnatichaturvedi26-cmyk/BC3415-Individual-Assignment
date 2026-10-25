"""PartCheck runtime helpers: finding / downloading the trained models, example pictures, loading customer pictures.

No web framework in here (so it can be tested without Streamlit). Settings come from environment variables:

    ARTIFACTS        folder that already holds the trained models (skips the download)
    MODEL_REPO       Hugging Face repo with the trained models (default: unnati002/review-triage-files)
    MODEL_REVISION   a branch, tag or commit of that repo, to pin the exact models (default: main)
    PARTCHECK_CACHE  where downloaded models are kept (default: ./_models, or the system temp folder if that is read-only)
    HF_TOKEN         only needed if the model repo is private
    MAX_ROWS         most reviews analysed per run on the dashboard (default 300 on CPU, 500 with a GPU)
    PARTCHECK_RULES_ONLY  1 = find problem types with the keyword rules only (no zero-shot language model); 0 = also use it.
                     Default: 1 on a server without a GPU (fast, small memory, and both screens then give the same answer), 0 with a GPU.
"""
import io
import json
import logging
import os
import tempfile

from PIL import Image, ImageDraw, ImageOps

log = logging.getLogger("partcheck")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO = os.environ.get("MODEL_REPO", "unnati002/review-triage-files")
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None
MARKER = "issue_config.json"
MAX_IMAGE_SIDE = 1280          # bigger pictures are shrunk: CLIP looks at 224 x 224 anyway, and a 12-megapixel photo wastes memory


class ModelsUnavailable(RuntimeError):
    """The trained models could not be found or downloaded. The message is written for the person using the app."""


def has_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def rules_only():
    v = os.environ.get("PARTCHECK_RULES_ONLY", "").strip().lower()
    return v in ("1", "true", "yes", "on") if v else not has_gpu()


def max_rows():
    try:
        return int(os.environ["MAX_ROWS"])
    except (KeyError, ValueError):
        return 500 if has_gpu() else 300


def _writable_dir(preferred):
    for d in (preferred, os.path.join(tempfile.gettempdir(), "partcheck")):
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    raise ModelsUnavailable("No writable folder was found to store the models. Set PARTCHECK_CACHE to a folder you can write to.")


def cache_dir():
    return _writable_dir(os.environ.get("PARTCHECK_CACHE") or os.path.join(ROOT, "_models"))


def artifacts_dir():
    """Folder holding sentiment/, image/ and issue_config.json. Looks locally first, then downloads once from Hugging Face."""
    candidates = [os.environ.get("ARTIFACTS"), os.path.join(ROOT, "artifacts"), os.path.join(os.environ.get("PARTCHECK_CACHE") or os.path.join(ROOT, "_models"), "artifacts"),
                  os.path.join(tempfile.gettempdir(), "partcheck", "artifacts")]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, MARKER)):
            return c
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ModelsUnavailable("The 'huggingface_hub' package is missing: run  pip install -r requirements.txt") from e
    log.info("Downloading the trained models from %s (first run only)...", MODEL_REPO)
    try:
        local = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=cache_dir(), token=os.environ.get("HF_TOKEN") or None,
                                  allow_patterns=["artifacts/*", "artifacts/**", "demo_auto_reviews.csv", "examples.json"])
    except Exception as e:
        log.exception("model download failed")
        raise ModelsUnavailable(f"The trained models could not be downloaded from the Hugging Face repo '{MODEL_REPO}' ({type(e).__name__}). "
                                "Check the internet connection and the MODEL_REPO setting, or copy the 'artifacts' folder next to this app.") from e
    art = os.path.join(local, "artifacts")
    if not os.path.exists(os.path.join(art, MARKER)):
        raise ModelsUnavailable(f"The repo '{MODEL_REPO}' was downloaded but has no artifacts/{MARKER}. Is MODEL_REPO correct?")
    return art


def load_models():
    """Loads everything the screens need. Slow (seconds to a minute): the app caches the result."""
    import triage_core as tc
    import triage_ui as ui
    m = ui.Models.load(artifacts_dir())
    if rules_only():                       # same settings the dashboard uses on a CPU server: no 700 MB zero-shot model, and both screens agree
        m.cfg = {k: {"method": "rules", "thr": 1.0} for k in tc.ISSUES}
    return m


def find_file(name):
    """A data file shipped with the app (demo reviews, example list), wherever it ended up."""
    folders = [ROOT, os.environ.get("ARTIFACTS") or "", os.path.join(os.environ.get("PARTCHECK_CACHE") or os.path.join(ROOT, "_models"))]
    for base in folders:
        p = os.path.join(base, name) if base else ""
        if p and os.path.exists(p):
            return p
    return None


def demo_rows():
    """Number of reviews in the demo file (the button label is built from it instead of a hard-coded number)."""
    p = find_file("demo_auto_reviews.csv")
    if not p:
        return 0
    with open(p, encoding="utf-8-sig") as f:
        import csv
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


# ---------------------------------------------------------------- pictures
def load_picture(file_or_bytes):
    """Opens a customer picture. Applies the phone's rotation flag (EXIF), so a sideways phone photo is not misread, and shrinks huge pictures.
    Returns an RGB PIL image. Raises ValueError with a readable message if the file is not a picture."""
    try:
        data = file_or_bytes.getvalue() if hasattr(file_or_bytes, "getvalue") else file_or_bytes
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        im.load()
    except Exception as e:
        raise ValueError("This file could not be opened as a picture. Use a JPG, PNG or WebP image.") from e
    if im.mode in ("RGBA", "LA", "P"):                     # transparent pictures: put them on white, not black
        rgba = im.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.split()[-1])
        im = base
    else:
        im = im.convert("RGB")
    im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return im


def _example_dir():
    return _writable_dir(os.path.join(cache_dir(), "examples_cache"))


def example_picture(name):
    """Path of an example picture, or None. Catalogue photos listed in examples.json are downloaded on first use (not redistributed);
    the 'random screenshot' is drawn locally. Never raises: a missing example just means its button is hidden."""
    try:
        d = _example_dir()
        path = os.path.join(d, name + (".png" if name == "random_screenshot" else ".jpg"))
        if os.path.exists(path):
            return path
        if name == "random_screenshot":
            im = Image.new("RGB", (480, 360), (245, 245, 247))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 480, 40], fill=(60, 120, 220))
            for i, t in enumerate(["Messages", "Hey, are we still on for lunch?", "Sure, 12:30 at the usual place", "Great, see you there!", "Running 5 min late", "No problem"]):
                dr.text((20, 60 + i * 40), t, fill=(30, 30, 30))
            im.save(path)
            return path
        items = json.load(open(find_file("examples.json") or os.path.join(ROOT, "examples.json"), encoding="utf-8"))
        item = next((it for it in items if it["name"] == name), None)
        if item is None:
            return None
        import requests
        r = requests.get(item["url"], timeout=10, headers={"User-Agent": "Mozilla/5.0 (PartCheck example fetcher)"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((640, 640))
        im.save(path, quality=88)
        return path
    except Exception:
        log.warning("example picture %s is not available", name, exc_info=True)
        return None
