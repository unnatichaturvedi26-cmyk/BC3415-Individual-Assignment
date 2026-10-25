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
from matplotlib.figure import Figure
from PIL import Image

import triage_core as tc
import triage_engine as te
import triage_image as ti
import triage_insights as ins

def _subplots(nrows=1, ncols=1, figsize=(6.4, 4.8)):
    """Like plt.subplots, but the figure is not registered in pyplot: safe with several visitors at once, and freed as soon as it is dropped."""
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols)


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
:root{--ink:#202b38;--muted:#687584;--line:#e2e7eb;--orange:#2a596d;--red:#a44443;--green:#29735e;--blue:#2a596d;--bg:#f9fafb;--card:#fff}
.stApp{background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif}
.stApp .block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.stApp h1,.stApp h2,.stApp h3{color:var(--ink);letter-spacing:-.025em}
.stApp [data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.stApp [data-testid="stBaseButton-primary"]{background:#233f52;border:1px solid #233f52;border-radius:5px;color:white}
.stApp [data-testid="stBaseButton-primary"] p{color:white;font-weight:600}
.stApp [data-testid="stBaseButton-primary"]:hover{background:#33586d;border-color:#33586d}
.stApp [data-testid="stButton"] button:not([kind="primary"]),.stApp [data-testid="stDownloadButton"] button{border-radius:5px;border-color:#cbd4db;background:#fff;color:var(--ink)}
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"]{border-bottom:1px solid var(--line);gap:24px}
.stApp [data-testid="stTabs"] button[role="tab"]{border-radius:0;font-weight:600;padding:12px 4px}
.stApp [data-testid="stTextArea"] textarea,.stApp [data-testid="stTextInput"] input,.stApp [data-testid="stSelectbox"]>div{border-radius:5px}
.pc-card,.pc-kpi,.pc-row,.pc-panel{color:var(--ink)}
.pc-hero{border-bottom:1px solid var(--line);padding:12px 0 20px;margin:0 0 14px}
.pc-eyebrow{font-size:10px;letter-spacing:.17em;color:#7b8794;font-weight:700;margin-bottom:8px}
.pc-hero h1{margin:0;font-size:34px;font-weight:700;letter-spacing:-.055em;line-height:1.12}
.pc-hero .sub{color:var(--muted);font-size:14px;margin:8px 0 0}
.pc-h2{font-size:20px;font-weight:700;letter-spacing:-.02em;margin:10px 0 5px}
.pc-sub{margin:0 0 17px;font-size:14px;color:var(--muted);line-height:1.55}
.pc-empty{padding:32px 20px;text-align:center;color:var(--muted);background:#fff;border:1px solid var(--line);border-radius:6px;font-size:14px}
.pc-banner{border-radius:5px;padding:10px 14px;margin:8px 0;font-size:14px;line-height:1.5;border:1px solid}
.pc-banner.info{background:#eff5f8;border-color:#c7dce8;color:#24475b}.pc-banner.ok{background:#ecf5f1;border-color:#c4ded3;color:#245d4b}
.pc-banner.warn{background:#fff8ec;border-color:#ead9b9;color:#795525}.pc-banner.err{background:#fff1ef;border-color:#edcbc6;color:#8c3836}
.pc-card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:16px 18px}
.pc-reco{display:flex;gap:14px;align-items:flex-start;border-left:3px solid;border-radius:4px;padding:13px 16px;margin-bottom:12px}
.pc-reco.flag{background:#fff8ec;border-color:#b7894c;color:#795525}.pc-reco.clear{background:#ecf5f1;border-color:var(--green);color:#245d4b}
.pc-reco.unknown{background:#f1f4f6;border-color:#8896a5;color:var(--ink)}
.pc-ico{width:32px;height:32px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:17px;flex:none;background:rgba(255,255,255,.75)}
.pc-rv{font-size:20px;font-weight:700;line-height:1.2}.pc-rs{font-size:14px;margin-top:3px;font-weight:500}
.pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pc-comb{grid-column:1/-1}
.pc-comb.high{border-color:#d8b1ac;background:#fff5f3}.pc-comb.medium{border-color:#ecd9b4;background:#fffaf1}.pc-comb.low{border-color:#c8dfd5;background:#f2f8f5}
.pc-panel{border:1px solid var(--line);border-radius:5px;padding:12px;background:#fff}
.pc-ph{font-size:12px;font-weight:700;color:var(--muted);margin-bottom:6px}
.pc-k{font-size:12px;color:var(--muted);margin-top:6px}.pc-v{font-size:15px;font-weight:600;color:var(--ink);line-height:1.3}.pc-n{font-size:12px;color:var(--muted)}
.pc-thumb{width:56px;height:56px;object-fit:cover;border-radius:4px;border:1px solid var(--line);float:right;margin-left:8px}
.pc-next{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;font-size:14px;line-height:1.5}
.pc-note{font-size:12px;color:var(--muted);margin-top:8px}
.pc-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}
.pc-kpi{background:#fff;border:1px solid var(--line);border-radius:5px;padding:15px}
.pc-kpi .n{font-size:27px;font-weight:700;line-height:1.15;color:var(--ink)}.pc-kpi .l{font-size:13px;color:var(--ink)}.pc-kpi .s{font-size:12px;color:var(--muted);margin-top:4px}
.pc-row{border:1px solid var(--line);border-radius:5px;margin:7px 0;background:#fff;overflow:hidden}
.pc-row summary{display:flex;align-items:center;gap:10px;padding:10px 13px;cursor:pointer;list-style:none;flex-wrap:wrap}
.pc-row summary::-webkit-details-marker{display:none}.pc-row summary:hover{background:#f7f9fa}
.pc-row .tx{flex:1;min-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}
.pc-row .body{padding:9px 14px 12px;border-top:1px solid var(--line);font-size:14px;line-height:1.55}
.pill{display:inline-block;min-width:96px;text-align:center;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:700}
.pill.HIGH{background:#fff0ed;color:var(--red)}.pill.MEDIUM{background:#fff6e6;color:#8e662b}.pill.LOW{background:#eff2f4;color:#52606e}
.chip{display:inline-block;padding:1px 8px;margin:1px 3px 1px 0;border-radius:4px;border:1px solid #d1e0e8;background:#f2f7f9;color:#31566b;font-size:12px;font-weight:600}
.chip.part{border-color:#d1e0e8;background:#f3f7f9}.chip.neg{border-color:#e8c8c2;background:#fff1ef;color:#8c3836}.chip.neu{border-color:#dde2e5;background:#f3f5f6;color:#586574}.chip.pos{border-color:#c8dfd5;background:#eff7f3;color:#245d4b}
@media(max-width:760px){.pc-grid{grid-template-columns:1fr 1fr}.pc-hero h1{font-size:30px}}
@media(max-width:500px){.pc-grid{grid-template-columns:1fr}}
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
        return {"state": "cannot", "note": "Add the product ordered to check whether the image matches it"}
    about = "the order is for" if source == "ordered" else "the review is about"
    shows = ti.FAMILY_NAME.get(photo["family"], photo["shows_name"])
    covered = []                                        # (part name, probability that the picture shows it); a part no picture family covers is skipped
    for part in parts:
        fams = ti.FAMILIES_FOR_KEYS.get(part["key"])
        if fams:
            covered.append((part["name"].lower(), sum(photo["families"].get(f, 0.0) for f in fams)))
    if not covered:
        return {"state": "cannot", "note": f"no picture family covers {parts[0]['name'].lower()}"}
    best_name, best_p = max(covered, key=lambda x: x[1])
    if best_p >= T_MATCH:                               # the picture matches ANY part the review names
        return {"state": "match", "note": f"{about} {best_name}; the picture looks like {shows}"}
    if best_p < T_MISMATCH:                             # ... and it is a mismatch only when it matches NONE of them
        return {"state": "mismatch", "note": f"{about} {covered[0][0]}, but the picture looks like {shows}"}
    return {"state": "cannot", "note": "the picture does not clearly show that part or a different one"}


# ---------------------------------------------------------------- screen 1: one order
RECO = {"flag": ("Review recommended", "!"), "clear": ("No issue detected", "✓"), "unknown": ("Unable to assess", "?")}
IMAGE_FLAG = {"wrong_item": "Wrong item", "defect": "Possible defect", "not_part": "Not an auto part", "ok": "No problem found", "cannot": "Visual check incomplete", "identified": "Part identified"}


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
        return "Low", ["Text: no problem found", "Picture: part identified, but condition or order match is not verified"]
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
    elif photo.get("status") == "part":
        image_flag = "identified"
    else:
        image_flag = "cannot"
    names = _issue_names(row)
    issue = ", ".join(n.lower() for n in names) if names else ""
    level, why = combine(row["priority_text_only"] if row is not None else "", float(row["p_negative"]) if row is not None else None, flags, issue, image_cannot=(image_flag in ("cannot", "identified")))
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
        if res.get("image_flag") in ("cannot", "identified"):
            return "No issue found in the review. The picture was recognised, but its condition was not verified."
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
        rows = [("Visual assessment", IMAGE_FLAG[res["image_flag"]], "Recognition and damage verification are separate checks.")]
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
            rows.append(("Condition", {"possible defect": "Possible defect", "no defect found": "No defect found"}.get(cond, "Not assessed — unverified"),
                         "" if cond else "Damage detection was not validated for this image; this is not a failure to recognise the part"))
        if match is not None and st in ("part", "unclear", "package"):
            state = {"match": "Match", "mismatch": "Do not match", "cannot": "Needs product reference"}[match["state"]]
            rows.append(("Order vs picture" if res.get("ordered_parts") else "Review vs picture", state, match["note"]))
        elif match is None and st == "part" and not res.get("ordered_parts") and row is None:
            rows.append(("Review vs picture", "Not compared", "add a review or the product ordered to compare"))
        if res["image_flag"] == "defect" and photo.get("condition") != "possible defect":
            rows[0] = ("Visual assessment", IMAGE_FLAG["defect"], "unvalidated cue: the picture resembles damaged parts or packaging")
        pic = _panel("2. Image signal", rows, thumb)
    # 3. combined decision
    lvl = res["level"]
    explanation = {"High": "Escalate: strong text evidence or multiple independent warning signals.",
                   "Medium": "Review: at least one text or picture signal warrants attention.",
                   "Low": "Routine: no confirmed warning signal; unverified image checks are not proof of good condition."}[lvl]
    comb = _panel("3. Combined decision", [("Priority", lvl, explanation)] + [("Evidence" if i == 0 else "", w, "") for i, w in enumerate(res["why"])], cls="pc-comb " + lvl.lower())
    return (f"<div class='pc-card'>{head}<div class='pc-grid'>{rev}{pic}{comb}</div>"
            f"<div class='pc-next'><b>Next action:</b> {esc(next_action(res))}</div>"
            f"<div class='pc-note'>A person decides what happens next. This app does not send messages or issue refunds.</div></div>")


# ---------------------------------------------------------------- screen 2: many reviews
class InputError(ValueError):
    """A problem with the person's input that can be shown to them as it is."""


def _read_upload(src):
    """src = a path or a file-like object (a Streamlit upload has .name and .getvalue()). CSV in UTF-8 or Windows encodings, or Excel."""
    name = str(getattr(src, "name", src)).lower()
    data = src.getvalue() if hasattr(src, "getvalue") else None
    handle = lambda: io.BytesIO(data) if data is not None else src
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            return pd.read_excel(handle())
        last = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(handle(), encoding=enc)
                if df.shape[1] == 1:                                  # a semicolon- or tab-separated export (common from European Excel)
                    for sep in (";", "\t"):
                        alt = pd.read_csv(handle(), encoding=enc, sep=sep)
                        if alt.shape[1] > 1:
                            return alt
                return df
            except UnicodeDecodeError as e:
                last = e
        raise InputError(f"Could not read the text encoding of this file ({last}). Save it as CSV UTF-8 and try again.")
    except ImportError as e:
        raise InputError(f"This file type needs an extra package: {e}. Save it as .xlsx or CSV instead.")
    except pd.errors.EmptyDataError:
        raise InputError("The file is empty.")
    except InputError:
        raise
    except Exception as e:
        raise InputError(f"The file could not be read as a spreadsheet ({type(e).__name__}). Check that it is a real .csv, .xlsx or .xls file.")


def read_table(pasted, sheet_path, demo_path=None):
    if demo_path:
        return pd.read_csv(demo_path), ("Demo data: real Amazon Automotive reviews (not used for training) with their real star ratings, dates and product names. "
                                        "About 30 reviews per product, so product comparisons are a demonstration, not a finding.")
    if sheet_path is not None and sheet_path != "":
        return _read_upload(sheet_path), ""
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
    fig, ax = _subplots(figsize=(5.2, 2.6)); n = max(len(q), 1)
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
    fig, ax = _subplots(figsize=(6.4, 2.8)); _style(ax)
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
    fig, ax = _subplots(figsize=(6.4, max(2.6, 0.38 * len(g) + 1))); _style(ax)
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
    fig, axes = _subplots(1, 2 if len(pt) else 1, figsize=(9.6, 3.2)); axes = np.atleast_1d(axes)
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


EXPORT_COLS = ["order_id", "product", "date", "rating", "review", "sentiment", "p_negative", "confirmed", "part", "evidence", "priority", "suggested_action"]


def _export_frame(q):
    cols = [c for c in EXPORT_COLS if c in q.columns]
    return q[cols].rename(columns={"p_negative": "chance_unhappy", "confirmed": "issues_found"})


def csv_bytes(q):
    """The export as UTF-8 bytes with a BOM, so Excel shows accents and quotes correctly. Nothing is written to disk."""
    return _export_frame(q).to_csv(index=False).encode("utf-8-sig")


def export_csv(q, path=None):
    """Writes the export to `path` (default: a private temporary file, never a shared name) and returns the path."""
    if path is None:
        import tempfile
        fd, path = tempfile.mkstemp(prefix="partcheck_", suffix=".csv")
        os.close(fd)
    _export_frame(q).to_csv(path, index=False)
    return path

# Apple-inspired editorial restraint: neutral type, white surfaces, subtle contrast, no gradients or decorative AI motifs.
CSS += """
:root{--ink:#1d1d1f;--muted:#6e6e73;--line:#e5e5e7;--bg:#f5f5f7;--green:#236c4d;--red:#a5413b}
.stApp{background:#f5f5f7;color:#1d1d1f;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',Arial,sans-serif}
.stApp .block-container{max-width:1260px;padding:32px 36px 90px}
.pc-hero{border:0;margin:0;padding:32px 0 34px;text-align:center}
.pc-eyebrow{letter-spacing:.095em;font-weight:600;font-size:11px;color:#737379}
.pc-hero h1{font-size:clamp(46px,6vw,72px);font-weight:700;letter-spacing:-.067em;line-height:1.08}
.pc-hero .sub{font-size:18px;letter-spacing:-.018em;margin-top:13px}
.stApp [data-testid='stTabs'] [data-baseweb='tab-list']{justify-content:center;border-bottom:1px solid #dedee1;gap:38px;margin-bottom:27px}
.stApp [data-testid='stTabs'] button[role='tab']{font-size:15px;font-weight:500;color:#6e6e73}
.stApp [data-testid='stBaseButton-primary']{border-radius:11px;background:#1d1d1f;border-color:#1d1d1f;min-height:44px}
.stApp [data-testid='stBaseButton-primary']:hover{background:#424245;border-color:#424245}
.stApp [data-testid='stButton'] button:not([kind='primary']),.stApp [data-testid='stDownloadButton'] button{border-radius:11px;min-height:43px;border:1px solid #dadade}
.stApp input,.stApp textarea,.stApp [data-baseweb='select']>div{border-radius:11px!important}
.pc-card,.pc-panel,.pc-kpi,.pc-row,.pc-empty{background:white;border:1px solid #e9e9eb;border-radius:17px;box-shadow:0 1px 2px rgba(0,0,0,.025)}
.pc-card{padding:23px}.pc-panel{padding:18px}.pc-kpi{padding:22px}.pc-kpi .n{font-size:32px;letter-spacing:-.045em}
.pc-h2{font-size:27px;font-weight:650;letter-spacing:-.045em;margin:11px 0 8px}.pc-sub{font-size:15px;color:#77777b}
.pc-reco{border-radius:13px;border-left:0;padding:19px;margin-bottom:20px}.pc-ico{border-radius:50%}
.pc-ph{font-size:12px;letter-spacing:.035em}.pc-v{font-size:15px}.pc-grid{gap:14px}
.pc-banner{border-radius:11px;padding:13px 17px}.pc-row{margin:10px 0}.pc-row summary{padding:16px}
.pill,.chip{border-radius:8px}.pc-next{margin-top:18px;padding-top:18px}
[data-testid='stFileUploader'] section{border-radius:13px;background:#fff;border-color:#dedee2}
[data-testid='stExpander']{background:white;border-radius:13px;overflow:hidden;border-color:#e5e5e7}
@media(max-width:760px){.stApp .block-container{padding:15px 17px 50px}.pc-hero{padding:22px 0}.pc-hero .sub{font-size:15px}.pc-grid{grid-template-columns:1fr}.pc-h2{font-size:22px}}
"""
