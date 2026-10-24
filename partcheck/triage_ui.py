"""PartCheck: the logic behind both screens (no web framework, so it can be tested without Gradio or a GPU).

Screen 1, CHECK AN ORDER.  A customer's review and/or one picture ->
    review : sentiment, the issue (damaged, does not fit ...), and the part the review names
    picture: is it an auto part at all?  which family of part does it show?  its condition, when that can be checked
    both   : does the picture match the part the review is about?
    -> one recommendation for a person.  The tool never sends messages or refunds anything.

Screen 2, SENTIMENT DASHBOARD.  Many reviews -> how customers feel (split, trend, by product), what drives the unhappy reviews
    (issue types, parts), and the orders a person should look at.
"""
import io
import os
import json
import html
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import triage_core as tc
import triage_engine as te
import triage_image as ti
import triage_insights as ins

LABEL = ins.LABEL
SENT = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
NAVY, ORANGE, RED, GREEN, GREY, BLUE, LINE = "#0F1A2B", "#FF9900", "#B12704", "#067D62", "#8896A5", "#146EB4", "#DAD4CA"

# ---- picture-check settings (measured in the notebook, Section 7d)
T_OTHER = 0.5        # the picture is "not an auto part" if CLIP puts at least this probability on non-part descriptions (food, pets, people, documents ...)
T_VEHICLE = 0.5      # ... or shows a whole vehicle / interior (context, not a part)
T_MISMATCH = 0.05    # the picture does not match the review if its probability of showing the review's part family is below this
T_DAMAGE = None     # unvalidated visible-damage cue: counts as a possible defect at or above this score (None = OFF until colab_picture_eval.py shows it is reliable)
T_PART = 0.50       # below this probability on 'an auto part' (all part classes together) the picture is treated as NOT an auto part
T_MATCH = 0.30       # ... and it matches if that probability is at least this (in between: cannot tell)

CSS = """
:root{--ink:#0F1A2B;--muted:#4A5868;--line:#DAD4CA;--orange:#FF9900;--red:#B12704;--green:#067D62;--blue:#146EB4;--bg:#FBF8F3;--card:#FFFFFF}
.gradio-container{background:var(--bg)!important;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif!important;max-width:1180px!important}
.gradio-container h1,.gradio-container h2,.gradio-container h3{color:var(--ink)!important;letter-spacing:-.01em}
.pc-comb.high{border-color:#B12704;background:#FDF1EE}.pc-comb.medium{border-color:#FF9900;background:#FFF7E8}.pc-comb.low{border-color:#067D62;background:#EDF8F4}
.pc-hero{background:#fff;border:1px solid var(--line);border-bottom:3px solid var(--orange);border-radius:12px;padding:14px 22px;margin:4px 0 10px 0}
.pc-hero h1{margin:0;font-size:26px;font-weight:750;display:inline-block}
.pc-hero .sub{color:var(--muted);font-size:15px;margin-left:14px}
.pc-h2{font-size:21px;font-weight:750;margin:2px 0 2px 0}
.pc-sub{margin:0 0 10px 0;font-size:14.5px;color:var(--muted)}
.pc-empty{padding:26px;text-align:center;color:var(--muted);background:var(--card);border:1px dashed var(--line);border-radius:12px;font-size:15px}
.pc-banner{border-radius:10px;padding:10px 14px;margin:8px 0;font-size:14.5px;line-height:1.5;border:1px solid}
.pc-banner.info{background:#EAF3FA;border-color:#B9D6EC;color:#0B3B63}.pc-banner.ok{background:#E7F5F0;border-color:#A8D8CA;color:#064D3C}
.pc-banner.warn{background:#FFF4E0;border-color:#F3CF8A;color:#7A3E00}.pc-banner.err{background:#FDECE8;border-color:#F0B7A8;color:#7F1D0B}
.pc-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px 12px 18px}
.pc-reco{display:flex;gap:14px;align-items:flex-start;border-left:5px solid;border-radius:10px;padding:13px 16px;margin-bottom:12px}
.pc-reco.flag{background:#FFF4E0;border-color:var(--orange);color:#7A3E00}.pc-reco.clear{background:#E7F5F0;border-color:var(--green);color:#064D3C}
.pc-reco.unknown{background:#EEF1F4;border-color:#8896A5;color:var(--ink)}
.pc-ico{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:17px;flex:none;background:rgba(255,255,255,.75)}
.pc-rv{font-size:21px;font-weight:750;line-height:1.2}.pc-rs{font-size:14.5px;margin-top:3px;font-weight:500}
.pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pc-comb{grid-column:1/-1}
.pc-panel{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#FFFEFC}
.pc-ph{font-size:13px;font-weight:700;color:var(--muted);margin-bottom:6px}
.pc-k{font-size:12.5px;color:var(--muted);margin-top:6px}.pc-v{font-size:15.5px;font-weight:650;color:var(--ink);line-height:1.3}.pc-n{font-size:12.5px;color:var(--muted)}
.pc-thumb{width:56px;height:56px;object-fit:cover;border-radius:8px;border:1px solid var(--line);float:right;margin-left:8px}
.pc-next{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;font-size:15px;line-height:1.5}
.pc-note{font-size:12.5px;color:var(--muted);margin-top:8px}
.pc-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:8px 0}
.pc-kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.pc-kpi .n{font-size:28px;font-weight:750;line-height:1.15;color:var(--ink)}.pc-kpi .l{font-size:13.5px;color:var(--ink)}.pc-kpi .s{font-size:12.5px;color:var(--muted);margin-top:2px}
.pc-row{border:1px solid var(--line);border-radius:10px;margin:7px 0;background:var(--card);overflow:hidden}
.pc-row summary{display:flex;align-items:center;gap:10px;padding:9px 13px;cursor:pointer;list-style:none;flex-wrap:wrap}
.pc-row summary::-webkit-details-marker{display:none}.pc-row summary:hover{background:#FAF6EF}
.pc-row .tx{flex:1;min-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14.5px}
.pc-row .body{padding:8px 14px 12px 14px;border-top:1px dashed var(--line);font-size:14.5px;line-height:1.55}
.pill{display:inline-block;min-width:96px;text-align:center;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:700}
.pill.HIGH{background:#FDECE8;color:var(--red)}.pill.MEDIUM{background:#FFF4E0;color:#C45500}.pill.LOW{background:#EEF1F4;color:#3D4B5C}
.chip{display:inline-block;padding:1px 9px;margin:1px 3px 1px 0;border-radius:6px;border:1px solid #B9D6EC;background:#EAF3FA;color:#0B3B63;font-size:12.5px;font-weight:600}
.chip.part{border-color:#CFE0EF;background:#F4F8FC}.chip.neg{border-color:#F0B7A8;background:#FDECE8;color:#7F1D0B}.chip.neu{border-color:#DAD4CA;background:#F3F1EC;color:#4A5868}.chip.pos{border-color:#A8D8CA;background:#E7F5F0;color:#064D3C}
@media (max-width:760px){.pc-grid{grid-template-columns:1fr}}
"""


def esc(s):
    return html.escape(str(s))


def banner(kind, text):
    return f"<div class='pc-banner {kind}'>{esc(text)}</div>"


def thumb_b64(img, size=112):
    im = img.copy().convert("RGB")
    im.thumbnail((size * 2, size * 2))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- models bundle
class Models:
    """sentiment model, issue-detector settings, the picture checks (None = no image model)."""

    def __init__(self, sentiment, cfg, image=None):
        self.sentiment, self.cfg, self.image = sentiment, cfg, image
        self.rec = ti.PhotoRecogniser(image.clip) if image is not None else None
        self.dmg = ti.DamageCues(image.clip) if image is not None else None

    @classmethod
    def load(cls, art):
        sent = te.load_sentiment(os.path.join(art, "sentiment"))
        cfg = json.load(open(os.path.join(art, "issue_config.json")))
        p = os.path.join(art, "image", "image_evidence.joblib")
        return cls(sent, cfg, ti.ImageEvidence.load(p) if os.path.exists(p) else None)


# ---------------------------------------------------------------- the picture
def look_at_photo(img, mdl):
    """What does the picture show?  status: 'part' | 'unclear' (a part, but not which) | 'other' (not an auto part) | 'package' | 'vehicle' | 'unavailable'."""
    if img is None:
        return None
    if mdl.rec is None:
        return {"status": "unavailable"}
    emb = mdl.rec.clip.embed_images([img])
    r = mdl.rec.predict(emb)[0]
    fam = max(r["families"], key=r["families"].get) if r["families"] else None
    out = {"status": "part", "shows": r["top"], "shows_name": ti.PHOTO_NAME[r["top"]], "family": fam, "families": r["families"],
           "other_name": ti.PHOTO_OTHER_NAME.get(r["other_top"], "something else"), "p_other": r["p_other"], "condition": None, "damage_score": None}
    if r["p_other"] >= T_OTHER or (r["p_part"] < T_PART and r["p_vehicle"] < T_VEHICLE):
        out["status"] = "package" if r["other_top"] == "package" else "other"
        if out["status"] == "package":
            out["shows_name"] = "a package"
    elif r["p_vehicle"] >= T_VEHICLE:
        out["status"] = "vehicle"
    elif not r.get("named", True):
        out["status"] = "unclear"
    elif mdl.image is not None:                       # validated condition check: only for factory-style photos of the seven part types it was built on
        out["condition"] = _condition(emb, mdl.image)
    dmg = getattr(mdl, "dmg", None)
    if dmg is not None and out["status"] in ("part", "unclear", "package"):
        out["damage_score"] = float(dmg.damage_score(emb)[0])
    return out


def _condition(emb, ev):
    g = ev.identify(emb)[0][0]
    if g is None or not bool(ev.known(emb, [g])[0]) or not ev.damage_checked(g):
        return None
    return "possible defect" if float(ev.damage_signal(emb, [g])[0]) >= te.PHOTO_FLAG else "no defect found"


def compare(parts, photo, source="review"):
    """Does the picture match the part the review (or the ordered product) is about?  -> {'state': 'match'|'mismatch'|'cannot', 'note': str} or None."""
    if photo is None:
        return None
    if photo["status"] == "other":
        return {"state": "cannot", "note": "the picture is not an auto part"}
    if photo["status"] == "package":
        return {"state": "cannot", "note": "the picture shows a package, not the part"}
    if photo["status"] == "unclear":
        return {"state": "cannot", "note": "no auto part can be clearly identified in the picture"}
    if photo["status"] != "part":
        return {"state": "cannot", "note": "the picture shows a whole vehicle, not a part" if photo["status"] == "vehicle" else "the picture model is not available"}
    if not parts:
        return {"state": "cannot", "note": "the review does not name a part"}
    fams = ti.FAMILIES_FOR_KEYS.get(parts[0]["key"])
    name = parts[0]["name"].lower()
    about = "the order is for" if source == "ordered" else "the review is about"
    if not fams:
        return {"state": "cannot", "note": f"no picture family covers {name}"}
    p = sum(photo["families"].get(f, 0.0) for f in fams)
    shows = ti.FAMILY_NAME.get(photo["family"], photo["shows_name"])
    if p >= T_MATCH:
        return {"state": "match", "note": f"{about} {name}; the picture looks like {shows}"}
    if p < T_MISMATCH:
        return {"state": "mismatch", "note": f"{about} {name}, but the picture looks like {shows}"}
    return {"state": "cannot", "note": "the picture does not clearly show that part or a different one"}


# ---------------------------------------------------------------- screen 1: one order
RECO = {"flag": ("Review recommended", "!"), "clear": ("No issue detected", "✓"), "unknown": ("Unable to assess", "?")}
IMAGE_FLAG = {"wrong_item": "Wrong item", "defect": "Possible defect", "not_part": "Not an auto part", "ok": "No problem found", "cannot": "Cannot assess"}


def _issue_names(row):
    return [LABEL.get(k, k) for k in _split(row["confirmed"])] if row is not None else []


def _split(s):
    return [k.strip() for k in str(s).split(",") if k.strip() and k.strip() != "-"]


def _first_phrase(row):
    ev = str(row["evidence"]) if row is not None else "-"
    return "" if ev in ("-", "") else ev.split("; ")[0].split(": ", 1)[-1].split("/")[0]


def combine(text_priority, p_neg, image_flags, text_issue="", image_cannot=False):
    """THE COMBINATION RULE.  Text evidence (0 none, 1 medium, 2 high) and picture evidence (0 none, 1 not an auto part, 2 wrong item or defect) -> High / Medium / Low.
    High: the text alone is high priority, OR text and picture both raise a problem, OR a wrong item / defect in the picture comes with an unhappy review.
    Medium: any one signal. Low: none.  Returns (level, [reasons])."""
    t = {"HIGH": 2, "MEDIUM": 1}.get(text_priority, 0)
    i = 2 if ("wrong_item" in image_flags or "defect" in image_flags) else (1 if "not_part" in image_flags else 0)
    why = []
    if t:
        why.append(f"Text: {text_issue or 'negative review'}")
    for f, msg in (("wrong_item", "Picture: shows a different part from the one the order or review is about"), ("defect", "Picture: possible visible damage or defect"),
                   ("not_part", "Picture: not an auto part")):
        if f in image_flags:
            why.append(msg)
    if t == 2 or (t >= 1 and i >= 1) or (i == 2 and p_neg is not None and p_neg >= 0.5):
        return "High", why
    if t or i:
        return "Medium", why
    if image_cannot:
        return "Low", ["Text: no problem found", "Picture: could not be assessed, so it adds no evidence"]
    return "Low", why or ["No problem found in the text or the picture"]


def assess_order(text, image, mdl, ordered=""):
    text, ordered = tc.fix_text(text or "").strip(), (ordered or "").strip()
    if not text and image is None:
        return {"error": "empty"}
    photo = look_at_photo(image, mdl) if image is not None else None
    damaged = bool(photo and (photo.get("condition") == "possible defect" or (photo.get("damage_score") is not None and T_DAMAGE is not None and photo["damage_score"] >= T_DAMAGE)))
    row, parts = None, []
    if text:
        row = te.triage([text], mdl.sentiment, mdl.cfg, image_risk=[1.0] if damaged else None, photo_kinds=["damage"] if damaged else None, sort=False).iloc[0]
        parts = tc.mentioned_parts(text)
    ordered_parts = tc.mentioned_parts(ordered) if ordered else []
    ref, source = (ordered_parts, "ordered") if ordered_parts else (parts, "review")
    match = compare(ref, photo, source) if (text or ordered_parts) else None
    text_flag = row is not None and row["priority_text_only"] in ("HIGH", "MEDIUM")
    not_part = bool(photo and photo["status"] == "other")
    mismatch = bool(match and match["state"] == "mismatch")
    if text:
        reco = "flag" if (text_flag or not_part or mismatch or damaged) else "clear"
    else:                                              # picture only
        reco = "flag" if (not_part or damaged) else ("clear" if (photo and photo.get("condition") == "no defect found") else "unknown")
    flags = [f for f, on in (("wrong_item", mismatch), ("defect", damaged), ("not_part", not_part)) if on]
    if photo is None:
        image_flag = None
    elif flags:
        image_flag = flags[0]
    elif match is not None and match["state"] == "match" or (photo.get("condition") == "no defect found"):
        image_flag = "ok"
    else:
        image_flag = "cannot"
    names = _issue_names(row)
    issue = ", ".join(n.lower() for n in names) if names else ""
    level, why = combine(row["priority_text_only"] if row is not None else "", float(row["p_negative"]) if row is not None else None, flags, issue, image_cannot=(image_flag == "cannot"))
    return {"text": text, "ordered": ordered, "image": image, "photo": photo, "row": row, "parts": parts, "ordered_parts": ordered_parts, "match": match, "reco": reco,
            "text_flag": text_flag, "not_part": not_part, "mismatch": mismatch, "damaged": damaged, "image_flag": image_flag, "level": level, "why": why}


def headline(res):
    """One line: what was found."""
    reco, row = res["reco"], res["row"]
    bits = []
    if res["text_flag"] and row is not None:
        names = _issue_names(row)
        part = (" with " + res["parts"][0]["name"].lower()) if res["parts"] else ""
        if len(names) == 1:
            bits.append(f"Possible {names[0].lower()} problem{part}")
        elif names:
            bits.append("Possible problems: " + ", ".join(n.lower() for n in names) + part)
        else:
            bits.append("Negative review with no specific problem named")
    if res["not_part"]:
        bits.append(f"The picture is not an auto part (it looks like {res['photo']['other_name']})")
    if res["mismatch"]:
        bits.append("The picture does not match the " + ("ordered product" if res.get("ordered_parts") else "review"))
    if res["damaged"]:
        bits.append("Possible defect in the picture" + (f" ({res['photo']['shows_name']})" if res["photo"]["status"] == "part" else ""))
    if bits:
        return "; ".join(bits)
    if reco == "clear":
        if res.get("image_flag") == "cannot":
            return "Nothing in the review points to a problem. The picture could not be assessed."
        return "Nothing in the review or the picture points to a problem."
    if res["photo"] is not None and not res["text"]:
        p = res["photo"]
        return {"vehicle": "The picture shows a whole vehicle, not a part.", "unavailable": "The picture model is not available.",
                "unclear": "No auto part can be clearly identified in the picture.", "package": "The picture shows a package; the part itself is not visible."}.get(
            p["status"], f"The picture shows {p.get('shows_name', 'a part')}; its condition cannot be assessed.")
    return "Not enough to assess."


def next_action(res):
    row = res["row"]
    if res["reco"] == "clear":
        return "No action needed."
    if res["reco"] == "unknown":
        return "Ask the customer for a clear photo of the part, or look at it by hand."
    acts = []
    if res["text_flag"] and row is not None:
        conf = _split(row["confirmed"])
        if "size_fit" in conf:
            acts.append("Check the part number against the vehicle's fitment data; offer an exchange for the right part (a photo cannot confirm fitment).")
        elif "damaged_defective" in conf and "delivery_shipping" in conf:
            acts.append("Offer a replacement or refund; review the packaging and the carrier (the part may have been damaged in transit).")
        else:
            first = str(row["suggested_action"]).split(" | ")[0]
            acts.append(first[len("Photo: "):] if first.startswith("Photo: ") else first)
    if res["not_part"]:
        acts.append("Ask the customer to send a photo of the part.")
    if res["mismatch"]:
        if res.get("ordered_parts"):
            acts.append("Check what was picked and shipped against the order; if the wrong part went out, send the right one and arrange the return")
        else:
            acts.append("Ask the customer which part the picture shows: it may belong to another order.")
    if res["damaged"] and not res["text_flag"]:
        acts.append(tc.ACTIONS["damaged_defective"])
    acts = [a.rstrip(".") + "." for a in acts]
    return " ".join(acts) if acts else "Look at the order by hand."


def _panel(title, rows, extra="", cls=""):
    body = "".join((f"<div class='pc-k'>{esc(k)}</div>" if k else "") + f"<div class='pc-v'>{esc(v)}</div>" + (f"<div class='pc-n'>{esc(n)}</div>" if n else "") for k, v, n in rows)
    return f"<div class='pc-panel {cls}'>{extra}<div class='pc-ph'>{esc(title)}</div>{body}</div>"


def render_order(res):
    if res.get("error"):
        return "<div class='pc-empty'>Add a review, a picture, or both, then press <b>Analyse order</b>.</div>"
    row, photo, parts, match = res["row"], res["photo"], res["parts"], res["match"]
    title, ico = RECO[res["reco"]]
    head = (f"<div class='pc-reco {res['reco']}'><div class='pc-ico'>{ico}</div><div><div class='pc-rv'>{esc(title)}</div>"
            f"<div class='pc-rs'>{esc(headline(res))}</div></div></div>")
    # 1. text signal
    if row is not None:
        names = _issue_names(row)
        poss = _split(row["possible"])
        rev = _panel("1. Text signal", [("Sentiment", f"{SENT.get(row['sentiment'], row['sentiment'])}", f"{row['p_negative']:.0%} chance the customer is unhappy"),
                                        ("Issue", ", ".join(names) if names else "No specific issue detected",
                                         (f"phrase in the review: “{_first_phrase(row)}”" + ("" if res["text_flag"] else ". Not escalated: the review is not negative overall")) if names and _first_phrase(row) else ("unconfirmed model guess: " + ", ".join(LABEL.get(k, k).lower() for k in poss) if poss else "")),
                                        ("Part named", ", ".join(p["name"] for p in parts[:2]) if parts else "No specific part named", "")])
    else:
        rev = _panel("1. Text signal", [("", "No review added", "")])
    # 2. image signal (its own flag)
    thumb = f"<img class='pc-thumb' src='data:image/jpeg;base64,{thumb_b64(res['image'])}' alt='picture'/>" if res["image"] is not None else ""
    if photo is None:
        pic = _panel("2. Image signal", [("", "No picture added", "")])
    else:
        rows = [("Image flag", IMAGE_FLAG[res["image_flag"]], "")]
        st = photo["status"]
        if st == "unavailable":
            rows = [("", "Unable to assess", "the picture model is not loaded")]
        elif st == "other":
            rows.append(("Shows", "Not an auto part", f"it looks like {photo['other_name']}"))
        elif st == "vehicle":
            rows.append(("Shows", "A whole vehicle or interior", "context, not a part"))
        elif st == "package":
            rows.append(("Shows", "A package or box", "the part itself is not visible"))
        elif st == "unclear":
            rows.append(("Shows", "No part clearly identified", "the picture could not be matched to one auto part"))
        else:
            cond = photo["condition"]
            rows.append(("Shows", photo["shows_name"].capitalize(), "an auto part"))
            rows.append(("Condition", {"possible defect": "Possible defect", "no defect found": "No defect found"}.get(cond, "Not assessed"),
                         "" if cond else "the validated damage check covers only factory-style photos of 7 part types"))
        if match is not None and st in ("part", "unclear", "package"):
            state = {"match": "Match", "mismatch": "Do not match", "cannot": "Cannot compare"}[match["state"]]
            rows.append(("Order vs picture" if res.get("ordered_parts") else "Review vs picture", state, match["note"]))
        elif match is None and st == "part" and not res.get("ordered_parts") and row is None:
            rows.append(("Review vs picture", "Not compared", "add a review or the product ordered to compare"))
        if res["image_flag"] == "defect" and photo.get("condition") != "possible defect":
            rows[0] = ("Image flag", IMAGE_FLAG["defect"], "unvalidated cue: the picture resembles damaged parts or packaging")
        pic = _panel("2. Image signal", rows, thumb)
    # 3. combined decision
    lvl = res["level"]
    comb = _panel("3. Combined decision", [("Priority", lvl, "")] + [("Based on" if i == 0 else "", w, "") for i, w in enumerate(res["why"])], cls="pc-comb " + lvl.lower())
    return (f"<div class='pc-card'>{head}<div class='pc-grid'>{rev}{pic}{comb}</div>"
            f"<div class='pc-next'><b>Next action:</b> {esc(next_action(res))}</div>"
            f"<div class='pc-note'>A person decides what happens next. This app does not send messages or issue refunds.</div></div>")


# ---------------------------------------------------------------- screen 2: many reviews
def read_table(pasted, sheet_path, demo_path=None):
    if demo_path:
        return pd.read_csv(demo_path), ("Demo data: real Amazon Automotive reviews (not used for training) with their real star ratings, dates and product names. "
                                        "About 30 reviews per product, so product comparisons are a demonstration, not a finding.")
    if sheet_path:
        df = pd.read_excel(sheet_path) if str(sheet_path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(sheet_path)
        return df, ""
    if pasted and pasted.strip():
        return pd.DataFrame({"review": [ln.strip() for ln in pasted.splitlines() if ln.strip()]}), ""
    raise ValueError("Upload a spreadsheet, paste some reviews, or try the demo data.")


def analyse_reviews(df, mdl, fast=False, max_rows=500):
    notes = []
    d = ins.standardise(df)
    if d.empty:
        raise ValueError("No reviews found. Check that your file has a column with the review text.")
    if len(d) > max_rows:
        notes.append(f"This server analyses the first {max_rows} reviews at a time ({len(d)} were provided).")
        d = d.head(max_rows).reset_index(drop=True)
    q = ins.analyse(d, mdl.sentiment, mdl.cfg, fast=fast)
    parts = [tc.mentioned_parts(t) for t in q["review"]]
    q["part"] = [p[0]["name"] if p else "" for p in parts]
    q["flagged"] = q["priority"].isin(["HIGH", "MEDIUM"])
    return {"q": q, "notes": notes}


def _pct(x):
    return f"{x:.0%}"


def kpis(q):
    n = len(q)
    neg, neu, pos = (int((q.sentiment == s).sum()) for s in ("negative", "neutral", "positive"))
    out = {"n": n, "neg": neg, "neu": neu, "pos": pos, "net": (pos - neg) / max(n, 1), "flagged": int(q.flagged.sum())}
    if "rating" in q.columns and q.rating.notna().any():
        out["avg_rating"] = float(q.rating.mean())
        out["hidden"] = int(((q.rating >= 4) & (q.sentiment == "negative")).sum())
    return out


def render_kpis(q):
    k = kpis(q)
    card = lambda n, l, s="": f"<div class='pc-kpi'><div class='n'>{n}</div><div class='l'>{l}</div><div class='s'>{s}</div></div>"
    cards = [card(k["n"], "reviews analysed", f"average rating {k['avg_rating']:.1f} stars" if "avg_rating" in k else ""),
             card(_pct(k["neg"] / max(k["n"], 1)), "negative", f"{k['neg']} reviews"),
             card(f"{k['net']:+.0%}", "net sentiment", "share positive minus share negative"),
             card(k["flagged"], "flagged for review", f"{_pct(k['flagged'] / max(k['n'], 1))} of reviews")]
    if "hidden" in k:
        cards.append(card(k["hidden"], "hidden unhappy customers", "4 or 5 stars, but the text is negative"))
    return "<div class='pc-kpis'>" + "".join(cards) + "</div>"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=NAVY)


def fig_split(q):
    fig, ax = plt.subplots(figsize=(5.2, 2.6)); n = max(len(q), 1)
    vals = [(q.sentiment == s).sum() for s in ("positive", "neutral", "negative")]
    bars = ax.barh(["Positive", "Neutral", "Negative"], vals, color=[GREEN, GREY, RED]); _style(ax); ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + n * 0.01, b.get_y() + b.get_height() / 2, f"{v} ({v / n:.0%})", va="center", color=NAVY, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.3 if max(vals) else 1); ax.set_title("How customers feel", loc="left", color=NAVY, fontweight="bold"); fig.tight_layout()
    return fig


def fig_trend(q):
    if "date" not in q.columns or q.date.notna().sum() < 12:
        return None
    d = q.dropna(subset=["date"]).copy()
    freq = "M" if (d.date.max() - d.date.min()).days > 150 else "W"
    d["period"] = d.date.dt.to_period(freq).dt.start_time
    g = d.groupby("period").agg(n=("review", "size"), neg=("sentiment", lambda s: (s == "negative").mean()), pos=("sentiment", lambda s: (s == "positive").mean()))
    g = g[g.n >= 3]
    if len(g) < 3:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.8)); _style(ax)
    ax.plot(g.index, g.neg * 100, color=RED, marker="o", label="% negative"); ax.plot(g.index, g.pos * 100, color=GREEN, marker="o", label="% positive")
    ax.set_ylabel("% of reviews"); ax.legend(frameon=False, loc="best"); ax.set_title("Sentiment over time (periods with 3+ reviews)", loc="left", color=NAVY, fontweight="bold")
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def fig_products(q, min_n=5, top=10):
    if "product" not in q.columns or q["product"].dropna().nunique() < 2:
        return None
    g = q.dropna(subset=["product"]).groupby("product").agg(n=("review", "size"), neg=("sentiment", lambda s: (s == "negative").mean()))
    g = g[g.n >= min_n].sort_values("neg", ascending=False).head(top)
    if len(g) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.38 * len(g) + 1))); _style(ax)
    ax.barh([f"{p[:34]} (n={n})" for p, n in zip(g.index, g.n)], g.neg * 100, color=RED); ax.invert_yaxis(); ax.set_xlabel("% negative reviews")
    ax.set_title("Where unhappiness concentrates (by product)", loc="left", color=NAVY, fontweight="bold"); fig.tight_layout()
    return fig


def fig_drivers(q):
    neg = q[q.sentiment == "negative"]
    if len(neg) < 3:
        return None
    issue = {LABEL[k]: int(neg.confirmed.str.contains(k).sum()) for k in tc.ISSUES}
    issue["No specific issue named"] = int((neg.confirmed == "-").sum())
    issue = {k: v for k, v in sorted(issue.items(), key=lambda x: -x[1]) if v}
    pt = neg[neg.part != ""].part.value_counts().head(8)
    fig, axes = plt.subplots(1, 2 if len(pt) else 1, figsize=(9.6, 3.2)); axes = np.atleast_1d(axes)
    _style(axes[0]); axes[0].barh(list(issue), list(issue.values()), color=[GREY if k.startswith("No specific") else BLUE for k in issue]); axes[0].invert_yaxis()
    axes[0].set_title(f"What the {len(neg)} negative reviews describe", loc="left", color=NAVY, fontweight="bold", fontsize=11)
    if len(pt):
        _style(axes[1]); axes[1].barh(list(pt.index), list(pt.values), color=ORANGE); axes[1].invert_yaxis()
        axes[1].set_title("Parts named in negative reviews", loc="left", color=NAVY, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def render_queue(q, show="Needs review", limit=40):
    d = q if show == "Everything" else q[q.flagged]
    if d.empty:
        return banner("ok", "No reviews are flagged. Switch to Everything to see all reviews.")
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    d = d.assign(_o=d.priority.map(order)).sort_values(["_o", "priority_score"], ascending=[True, False]).head(limit)
    rows = []
    for _, r in d.iterrows():
        issues = "".join(f"<span class='chip'>{esc(LABEL.get(k, k))}</span>" for k in _split(r.confirmed)) or "<span class='chip neu'>No specific issue</span>"
        part = f"<span class='chip part'>{esc(r.part)}</span>" if r.part else ""
        sc = {"negative": "neg", "neutral": "neu", "positive": "pos"}[r.sentiment]
        meta = " &middot; ".join(esc(str(r[c])) for c in ("product", "order_id") if c in r.index and pd.notna(r[c]) and str(r[c]).strip() not in ("", "nan"))
        ev = r.evidence if r.evidence != "-" else "no matching phrase"
        act = str(r.suggested_action).split(" | ")[0]
        rows.append(f"<details class='pc-row'><summary><span class='pill {r.priority}'>{r.priority.capitalize()} priority</span><span class='chip {sc}'>{SENT[r.sentiment]}</span>"
                    f"{part}{issues}<span class='tx'>{esc(r.review[:140])}</span></summary><div class='body'><div>{esc(r.review)}</div>"
                    f"{('<div class=pc-n>' + meta + '</div>') if meta else ''}<div class='pc-n' style='margin-top:6px'>Evidence: {esc(ev)}</div>"
                    f"<div style='margin-top:6px'><b>Next action:</b> {esc(act)}</div></div></details>")
    return "".join(rows) + "<div class='pc-note'>Recommendations only: a person decides. Nothing is sent or refunded by this app.</div>"


def export_csv(q, path="partcheck_reviews.csv"):
    cols = [c for c in ["order_id", "product", "date", "rating", "review", "sentiment", "p_negative", "confirmed", "part", "evidence", "priority", "suggested_action"] if c in q.columns]
    q[cols].rename(columns={"p_negative": "chance_unhappy", "confirmed": "issues_found"}).to_csv(path, index=False)
    return path
