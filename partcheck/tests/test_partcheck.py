"""Logic tests for PartCheck (no GPU, no neural models): stand-in models with scripted pictures.
A solid-colour picture stands for a scenario:  red = a tyre, blue = a bottle of oil, yellow = floor mats, green = food (a random picture),
grey = a whole vehicle, orange = a damaged seat cover (factory-style), purple = a picture that could be either (unclear).
Run:  python tests/test_partcheck.py
"""
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from PIL import Image

import triage_core as tc
import triage_image as ti
import triage_engine as te
import triage_ui as ui

FAILS, PASSES = [], 0


def check(name, cond, detail=""):
    global PASSES
    if cond:
        PASSES += 1
        print("PASS  " + name)
    else:
        FAILS.append(name)
        print("FAIL  " + name + (f"   [{detail}]" if detail else ""))


def solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


RED, BLUE, YELLOW, GREEN, GREY, ORANGE, PURPLE = (200, 30, 30), (30, 30, 200), (200, 200, 30), (30, 200, 30), (128, 128, 128), (230, 120, 10), (150, 50, 150)
BROWN, TEAL = (120, 80, 40), (30, 150, 150)


class StubSentiment:
    NEG = {"leaking", "broke", "cracked", "disappointed", "terrible", "awful", "bad", "horrible", "not", "doesn't", "didn't", "never", "soaked", "worst", "hate", "returned", "stopped"}
    POS = {"great", "love", "perfect", "excellent", "good", "happy", "easy", "nice", "works", "thanks", "thank"}

    def _p(self, t):
        w = re.findall(r"[a-z']+", t.lower()); n = sum(x in self.NEG for x in w); p = sum(x in self.POS for x in w)
        neg = min(0.97, 0.05 + 0.3 * n) if n > p else 0.03
        pos = 0.9 if p > n else 0.1
        return np.array([neg, 1 - neg - (pos if p > n else 0.0) if p > n else (1 - neg) * 0.3, pos if p > n else (1 - neg) * 0.7])

    def predict_proba(self, texts):
        P = np.array([self._p(t) for t in texts]); return P / P.sum(1, keepdims=True)

    def predict_label(self, texts):
        return self.predict_proba(texts).argmax(1)


class StubClip:
    def embed_images(self, imgs):
        return np.array([np.array(i.resize((1, 1)).convert("RGB"))[0, 0] / 255.0 for i in imgs])


def _kind(e):
    r, g, b = (e * 255).round()
    if abs(r - 200) < 20 and abs(g - 30) < 20 and abs(b - 30) < 20: return "tire"
    if abs(r - 30) < 20 and abs(g - 30) < 20 and abs(b - 200) < 20: return "bottle"
    if abs(r - 200) < 20 and abs(g - 200) < 20 and abs(b - 30) < 20: return "floor_mat"
    if abs(r - 30) < 20 and abs(g - 200) < 20 and abs(b - 30) < 20: return "food"
    if abs(r - 128) < 20 and abs(g - 128) < 20 and abs(b - 128) < 20: return "vehicle"
    if abs(r - 230) < 20 and abs(g - 120) < 20 and abs(b - 10) < 20: return "seat_cover"
    if abs(r - 120) < 20 and abs(g - 80) < 20 and abs(b - 40) < 20: return "package"
    if abs(r - 30) < 20 and abs(g - 150) < 20 and abs(b - 150) < 20: return "mystery"
    return "unclear"


class StubRec:
    clip = StubClip()
    FAM = {"tire": {"wheels": 0.92, "mechanical": 0.08}, "bottle": {"fluids": 0.92, "tools": 0.08}, "floor_mat": {"interior": 0.92, "exterior": 0.08},
           "seat_cover": {"interior": 0.92, "exterior": 0.08}, "unclear": {"fluids": 0.15, "interior": 0.15, "wheels": 0.4, "tools": 0.3}}

    def predict(self, emb):
        out = []
        for e in emb:
            k = _kind(e)
            d = {"top": k if k in ("tire", "bottle", "floor_mat", "seat_cover") else "tire", "top_p": 0.9, "top3": [k], "p_part": 0.95, "p_other": 0.02, "p_vehicle": 0.02,
                 "share": 0.9, "families": dict(self.FAM.get(k, {"wheels": 1.0})), "other_top": "food"}
            if k == "food": d.update(p_part=0.03, p_other=0.95, other_top="food")
            if k == "package": d.update(p_part=0.05, p_other=0.9, other_top="package")
            if k == "mystery": d.update(named=False)
            if k == "vehicle": d.update(p_part=0.05, p_vehicle=0.9)
            out.append(d)
        return out


class StubEv:
    clip = StubClip()

    def identify(self, emb):
        return [("leather", 1.0) if _kind(e) == "seat_cover" else (None, 0.0) for e in emb]

    def known(self, emb, prods):
        return np.array([_kind(e) == "seat_cover" for e in emb])

    def damage_checked(self, p):
        return True

    def damage_signal(self, emb, prods):
        return np.array([1.0 for _ in emb])


class StubModels:
    sentiment = StubSentiment()
    cfg = {k: {"method": "rules", "thr": 1.0} for k in tc.ISSUES}
    image = StubEv()
    rec = StubRec()


M = StubModels()


def balanced(h):
    return h.count("<div") == h.count("</div>")


print("== 1. TEXT ONLY ==")
r = ui.assess_order("The bottle of motor oil arrived leaking and the box is soaked, very disappointed.", None, M); h = ui.render_order(r)
check("a damaged-oil review is flagged, names the issue and the part", r["reco"] == "flag" and "Review recommended" in h and "motor oil" in ui.headline(r).lower() and r["parts"][0]["name"] == "Motor oil")
check("...and the next action is about the damage, not about a picture", "replacement" in ui.next_action(r).lower() or "refund" in ui.next_action(r).lower())
r = ui.assess_order("Great product, works perfectly, thank you!", None, M)
check("a happy review is not flagged", r["reco"] == "clear" and "No issue detected" in ui.render_order(r))
r = ui.assess_order("It doesn't fit my 2018 F-150. The bolt holes are off.", None, M)
check("a fit complaint goes to the fitment data and says a picture cannot confirm fit", "fitment data" in ui.next_action(r) and "cannot confirm fitment" in ui.next_action(r))
r = ui.assess_order("Looks great and feels well made, but it doesn't fit my 2015 Civic.", None, M)
check("a happy-sounding review with a real fit problem is flagged (mixed complaint)", r["reco"] == "flag")
r = ui.assess_order("Works great! Rotate this between my Truck, UTV, Lawn tractor, and Motorcycle.", None, M)
check("a list of vehicles is not a part: no part named", r["parts"] == [] and "No specific part named" in ui.render_order(r))
check("nothing entered: a friendly prompt", "Analyse order" in ui.render_order(ui.assess_order("", None, M)))

print("\n== 2. PICTURE ONLY ==")
r = ui.assess_order("", solid(GREEN), M); h = ui.render_order(r)
check("a random picture (food) is flagged as 'not an auto part'", r["reco"] == "flag" and "Not an auto part" in h and "not an auto part" in ui.headline(r).lower())
check("...and the next action asks for a photo of the part", "photo of the part" in ui.next_action(r))
r = ui.assess_order("", solid(GREY), M)
check("a whole-vehicle picture is not a part: Unable to assess", r["reco"] == "unknown" and "Unable to assess" in ui.render_order(r))
r = ui.assess_order("", solid(RED), M); h = ui.render_order(r)
check("a tyre picture alone: says it shows tyres, and that the condition is not assessed (no false 'all clear')", r["reco"] == "unknown" and "Tyres" in h and "Not assessed" in h)
r = ui.assess_order("", solid(ORANGE), M)
check("a factory-style picture the damage check can read: possible defect is flagged", r["reco"] == "flag" and "Possible defect" in ui.render_order(r))

print("\n== 3. REVIEW + PICTURE: do they match? ==")
r = ui.assess_order("The bottle of motor oil arrived leaking.", solid(RED), M); h = ui.render_order(r)
check("YOUR CASE: a review about a bottle of oil with a picture of a tyre: the mismatch is flagged", r["mismatch"] and r["match"]["state"] == "mismatch" and "Do not match" in h)
check("...the headline and next action say so", "does not match" in ui.headline(r) and "which part the picture shows" in ui.next_action(r))
r = ui.assess_order("Great, thank you, love the motor oil.", solid(RED), M)
check("a happy oil review with a tyre picture is still flagged for the mismatch", r["reco"] == "flag" and r["mismatch"] and not r["text_flag"])
r = ui.assess_order("These floor mats are great, perfect fit.", solid(YELLOW), M)
check("a floor-mat review with a floor-mat picture: Match, not flagged", r["match"]["state"] == "match" and r["reco"] == "clear" and "Match" in ui.render_order(r))
r = ui.assess_order("Great product, thank you!", solid(GREEN), M)
check("a happy review with a random picture: flagged, and the comparison says it cannot compare", r["reco"] == "flag" and r["match"]["state"] == "cannot" and "not an auto part" in r["match"]["note"])
r = ui.assess_order("Great product, thank you!", solid(RED), M)
check("a review that names no part cannot be compared with the picture: no mismatch claimed", r["match"]["state"] == "cannot" and not r["mismatch"] and r["reco"] == "clear")
r = ui.assess_order("The motor oil is great.", solid(PURPLE), M)
check("an unclear picture is not called a mismatch either", r["match"]["state"] == "cannot" and not r["mismatch"])
r = ui.assess_order("Love it, looks great!", solid(ORANGE), M)
check("a happy review with a damaged-looking factory-style picture: flagged for the defect", r["reco"] == "flag" and r["damaged"] and "Possible defect" in ui.headline(r))
for im in (RED, GREEN, YELLOW, GREY, ORANGE):
    check(f"HTML is well formed (picture {im})", balanced(ui.render_order(ui.assess_order("The motor oil leaked.", solid(im), M))))

print("\n== 4. THE SENTIMENT DASHBOARD ==")
demo = os.path.join(ROOT, "demo_auto_reviews.csv")
df = pd.read_csv(demo) if os.path.exists(demo) else pd.DataFrame({"review": ["Arrived cracked, very disappointed", "Great, love it"] * 20, "date": pd.date_range("2023-01-01", periods=40, freq="7D"),
                                                                    "rating": [1, 5] * 20, "product": ["Alpha"] * 20 + ["Beta"] * 20})
res = ui.analyse_reviews(df, M, fast=True)
q = res["q"]; k = ui.kpis(q)
check("every review gets a sentiment, an issue column, a part column and a priority", {"sentiment", "confirmed", "part", "priority", "flagged"} <= set(q.columns) and len(q) == len(df))
check("the sentiment counts add up to the number of reviews", k["pos"] + k["neu"] + k["neg"] == k["n"])
check("net sentiment = share positive minus share negative", abs(k["net"] - (k["pos"] - k["neg"]) / k["n"]) < 1e-9)
kp = ui.render_kpis(q)
check("KPI cards: reviews, negative share, net sentiment, flagged", all(s in kp for s in ("reviews analysed", "negative", "net sentiment", "flagged for review")) and balanced(kp))
check("no invented dollar figures in the KPI cards (the review list may quote a customer's own price)", "$" not in kp)
check("sentiment split chart is drawn", ui.fig_split(q) is not None)
check("trend and by-product charts exist when dates and products exist (and are skipped when they do not)", (ui.fig_trend(q) is not None) == (q.date.notna().sum() >= 12) and
      ui.fig_trend(ui.analyse_reviews(pd.DataFrame({"review": ["bad", "good", "ok"] * 6}), M, True)["q"]) is None)
check("what-drives-unhappiness chart exists when there are negative reviews", ui.fig_drivers(q) is not None or (q.sentiment == "negative").sum() < 3)
qh = ui.render_queue(q, "Needs review"); check("the review list shows priorities and is well formed", ("pill" in qh or "No reviews are flagged" in qh) and balanced(qh))
out = ui.export_csv(q, os.path.join(tempfile.mkdtemp(), "x.csv")); back = pd.read_csv(out)
check("the CSV export has sentiment, issues, part and priority", {"sentiment", "issues_found", "part", "priority"} <= set(back.columns) and len(back) == len(q))
paste = ui.analyse_reviews(ui.read_table("Arrived cracked\nLove it\nnever worked", None)[0], M, True)
check("pasted reviews work (no other columns needed)", len(paste["q"]) == 3)
try:
    ui.analyse_reviews(pd.DataFrame({"review": ["", None]}), M, True); check("no reviews at all: a clear error", False)
except ValueError as e:
    check("no reviews at all: a clear error", "No reviews found" in str(e))

print("\n== 5. IMAGE FLAG, ORDERED PRODUCT, COMBINED DECISION ==")
r = ui.assess_order("Great, love it.", solid(GREEN), M)
check("random picture: the image flag is 'Not an auto part' and the priority is at least Medium", r["image_flag"] == "not_part" and r["level"] in ("Medium", "High"))
r = ui.assess_order("The oil leaked, terrible, disappointed.", solid(RED), M); h = ui.render_order(r)
check("negative review plus a picture of a different part: image flag Wrong item, combined priority High, reasons listed",
      r["image_flag"] == "wrong_item" and r["level"] == "High" and len(r["why"]) == 2 and "3. Combined decision" in h and "2. Image signal" in h and "1. Text signal" in h)
r = ui.assess_order("Great, thank you, love the motor oil.", solid(RED), M)
check("happy review plus wrong-part picture: Medium, not High", r["image_flag"] == "wrong_item" and r["level"] == "Medium")
r = ui.assess_order("Works great.", solid(RED), M, ordered="Valvoline synthetic motor oil 5 quart")
check("ORDERED product oil, picture of a tyre: wrong delivery is flagged against the order", r["mismatch"] and "Order vs picture" in ui.render_order(r) and "ordered product" in ui.headline(r) and "picked and shipped" in ui.next_action(r))
r = ui.assess_order("", solid(RED), M, ordered="Valvoline synthetic motor oil 5 quart")
check("an ordered product with a picture and no review can still be compared", r["image_flag"] == "wrong_item")
r = ui.assess_order("Works great.", solid(RED), M, ordered="Michelin all season tire")
check("ordered tyre, picture of a tyre: match, image flag 'No problem found'", r["image_flag"] == "ok" and r["level"] == "Low")
r = ui.assess_order("Works great.", solid(TEAL), M)
check("a picture the model cannot name is not given a part name", r["photo"]["status"] == "unclear" and r["image_flag"] == "cannot" and "No part clearly identified" in ui.render_order(r) and not r["mismatch"])
r = ui.assess_order("Box was crushed and the part inside is broken.", solid(BROWN), M)
check("a packaging photo is recognised as a package, not 'not an auto part'", r["photo"]["status"] == "package" and not r["not_part"] and "A package or box" in ui.render_order(r))
r = ui.assess_order("", None, M)
check("empty input still asks for input", r.get("error") == "empty")
check("combine(): no signal is Low", ui.combine("", 0.1, [])[0] == "Low")
check("combine(): text HIGH alone is High; one weak signal is Medium", ui.combine("HIGH", 0.9, [])[0] == "High" and ui.combine("MEDIUM", 0.6, [])[0] == "Medium")
check("combine(): text MEDIUM plus a picture problem is High", ui.combine("MEDIUM", 0.6, ["not_part"])[0] == "High")
check("part list: bare 'blades' is read as wiper blades", [p["key"] for p in tc.mentioned_parts("Install is pretty simple, these blades last longer.")] == ["wipers"])
for im in (RED, TEAL, BROWN, GREEN):
    check(f"HTML is well formed for the three-part card (picture {im})", balanced(ui.render_order(ui.assess_order("The motor oil leaked.", solid(im), M))))

print("\n== 6. CONSISTENCY OF THE PICTURE CLASSES ==")
keys_in_lexicon = {p[0] for p in tc.PARTS}
check("every picture class has a family", all(c[0] in ti.PHOTO_FAMILY for c in ti.PHOTO_CLASSES), str([c[0] for c in ti.PHOTO_CLASSES if c[0] not in ti.PHOTO_FAMILY]))
check("every lexicon key a picture class refers to exists in the part list", all(k in keys_in_lexicon for c in ti.PHOTO_CLASSES for k in c[3]), str({k for c in ti.PHOTO_CLASSES for k in c[3] if k not in keys_in_lexicon}))
check("every non-part class has a display name", all(k in ti.PHOTO_OTHER_NAME for k, _ in ti.PHOTO_NON_PART), str([k for k, _ in ti.PHOTO_NON_PART if k not in ti.PHOTO_OTHER_NAME]))
check("a steering wheel review names the part and the picture classes can show it", [p["key"] for p in tc.mentioned_parts("The steering wheel feels great")] == ["steering_wheel"] and "steering" in ti.PHOTO_FOR_KEYS["steering_wheel"])
check("wiper blades (exterior) and a steering wheel (interior) are different families", ti.PHOTO_FAMILY["steering"] != ti.PHOTO_FAMILY["wiper"])

print(f"\n{PASSES} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILED:", *FAILS, sep="\n  - ")
    sys.exit(1)

