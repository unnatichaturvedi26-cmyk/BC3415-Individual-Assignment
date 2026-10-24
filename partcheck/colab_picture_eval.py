"""PartCheck picture checks: one evaluation script to run on Colab (GPU, about 10 to 15 minutes).

It measures three things and prints the numbers to quote in the video and the notebook:
  1. Random pictures: how many are rejected as "not an auto part" (food, pets, flowers, and general photos with instruments, documents and so on).
  2. Real customer photos: how many are wrongly rejected, and how many cannot be given a part name.
  3. Visible-damage cue: AUC and a suggested threshold, using real customer photos from reviews that complain about damage versus happy reviews.

How to run on Colab (T4 GPU):
    !pip -q install sentence-transformers datasets huggingface_hub pillow requests scikit-learn
    from huggingface_hub import snapshot_download
    snapshot_download("unnati002/review-triage-files", local_dir="/content/partcheck")   # the repo made by upload_to_hf.py
    %cd /content/partcheck
    !python colab_picture_eval.py
(colab_picture_eval.py is in the upload folder; if it is not there, upload it next to triage_image.py.)
"""
import io
import itertools
import json
import os
import random
import sys

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.path.insert(0, os.getcwd())
import triage_core as tc
import triage_image as ti

random.seed(0)
clip = ti.ClipScorer()
rec = ti.PhotoRecogniser(clip)
dmg = ti.DamageCues(clip)
T_OTHER = 0.5
report = {}


def embed(images):
    return clip.embed_images(images)


# ------------------------------------------------------------------ 1. random pictures
def take(ds_name, split, field, n, **kw):
    from datasets import load_dataset
    ds = load_dataset(ds_name, split=split, streaming=True, **kw)
    out = []
    for r in itertools.islice(ds, n):
        im = r[field]
        if im is not None:
            out.append(im.convert("RGB"))
    return out


SETS = [("food101", "ethz/food101", "validation", "image", {}), ("cats_vs_dogs", "microsoft/cats_vs_dogs", "train", "image", {}),
        ("flowers", "nelorth/oxford-flowers", "test", "image", {}), ("tiny_imagenet", "zh-plus/tiny-imagenet", "valid", "image", {})]
print("\n== 1. RANDOM PICTURES (should be rejected as not an auto part) ==")
tot, rej, per = 0, 0, {}
for name, ds, split, field, kw in SETS:
    try:
        ims = take(ds, split, field, 150, **kw)
        pr = rec.predict(embed(ims))
        r = float(np.mean([p["p_other"] >= T_OTHER for p in pr]))
        per[name] = round(r, 3)
        tot += len(ims)
        rej += sum(p["p_other"] >= T_OTHER for p in pr)
        print(f"  {name:14s} n={len(ims):3d}  rejected as not-a-part: {r:.0%}")
    except Exception as e:  # a dataset may be unavailable: skip it and say so
        print(f"  {name:14s} skipped ({type(e).__name__})")
report["random_rejected"] = {"overall": round(rej / max(tot, 1), 3), "n": tot, "per_set": per}
print(f"  ALL            n={tot}  rejected: {rej / max(tot, 1):.0%}")

# ------------------------------------------------------------------ 2 + 3. real customer photos from Amazon Automotive reviews
print("\n== 2. REAL CUSTOMER PHOTOS (streamed from Amazon Reviews 2023, Automotive) ==")
from huggingface_hub import HfFileSystem

BAD, GOOD = [], []
with HfFileSystem().open("datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Automotive.jsonl", "r", encoding="utf-8") as h:
    for line in itertools.islice(h, 1_200_000):
        r = json.loads(line)
        imgs = [x.get("large_image_url") or x.get("medium_image_url") for x in (r.get("images") or []) if isinstance(x, dict)]
        imgs = [u for u in imgs if u]
        txt = (r.get("text") or "")
        if not imgs or len(txt) < 25:
            continue
        flags = tc.rule_flags([r.get("title", "") + " " + txt]).iloc[0]
        if flags["damaged_defective"] and r.get("rating", 3) <= 2 and len(BAD) < 600:
            BAD.append((imgs[0], txt))
        elif not flags.any() and r.get("rating", 3) >= 4 and len(GOOD) < 600:
            GOOD.append((imgs[0], txt))
        if len(BAD) >= 600 and len(GOOD) >= 600:
            break
print("  photo reviews found: damaged-complaint", len(BAD), "| happy, no issue", len(GOOD))


def fetch(u):
    try:
        x = requests.get(u, timeout=12)
        im = Image.open(io.BytesIO(x.content)).convert("RGB")
        im.thumbnail((448, 448))
        return im
    except Exception:
        return None


with ThreadPoolExecutor(24) as ex:
    bad_im = [i for i in ex.map(fetch, [u for u, _ in BAD]) if i is not None]
    good_im = [i for i in ex.map(fetch, [u for u, _ in GOOD]) if i is not None]
Eb, Eg = embed(bad_im), embed(good_im)
allE = np.vstack([Eb, Eg])
pr = rec.predict(allE)
n = len(pr)
print(f"  real photos: {n}")
print(f"  wrongly rejected as not-a-part: {np.mean([p['p_other'] >= T_OTHER for p in pr]):.1%}")
print(f"  shown as a whole vehicle:       {np.mean([p['p_vehicle'] >= 0.5 for p in pr]):.1%}")
print(f"  part could not be named (unclear): {np.mean([(not p['named']) and p['p_other'] < T_OTHER and p['p_vehicle'] < 0.5 for p in pr]):.1%}")
print(f"  packaging photos:               {np.mean([p['p_other'] >= T_OTHER and p['other_top'] == 'package' for p in pr]):.1%}")
report["real_photos"] = {"n": n, "wrongly_rejected": round(float(np.mean([p['p_other'] >= T_OTHER for p in pr])), 3),
                         "unclear": round(float(np.mean([(not p['named']) and p['p_other'] < T_OTHER and p['p_vehicle'] < 0.5 for p in pr])), 3)}

# ------------------------------------------------------------------ 3. visible-damage cue
print("\n== 3. VISIBLE-DAMAGE CUE (photos from damage complaints vs photos from happy reviews) ==")
from sklearn.metrics import roc_auc_score, precision_recall_curve

s = dmg.damage_score(allE)
y = np.r_[np.ones(len(Eb)), np.zeros(len(Eg))]
auc = roc_auc_score(y, s)
print(f"  AUC = {auc:.3f}   (0.5 = no better than chance; the damage-complaint photos are a noisy label, so a modest AUC is expected)")
P, R, T = precision_recall_curve(y, s)
base = y.mean()
best = None
for p_, r_, t_ in zip(P[:-1], R[:-1], T):
    if p_ >= 0.75 and r_ >= 0.10:
        best = (t_, p_, r_) if best is None or r_ > best[2] else best
print(f"  base rate of damage-complaint photos in this sample: {base:.0%}")
if best:
    print(f"  suggested T_DAMAGE = {best[0]:.2f}: precision {best[1]:.0%}, recall {best[2]:.0%}")
    report["damage_cue"] = {"auc": round(float(auc), 3), "T_DAMAGE": round(float(best[0]), 2), "precision": round(float(best[1]), 3), "recall": round(float(best[2]), 3)}
else:
    print("  no threshold reaches 75% precision: leave T_DAMAGE = None in triage_ui.py (switch the cue off) and say so in the report.")
    report["damage_cue"] = {"auc": round(float(auc), 3), "T_DAMAGE": None}

json.dump(report, open("picture_eval.json", "w"), indent=1)
print("\nSaved picture_eval.json. Send it (or paste the numbers above) back so the app threshold and the report can be set.")
