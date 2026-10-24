"""From reviews to decisions.

Turns triaged reviews into what a manager needs: what to fix first, where it concentrates (product / supplier),
whether it is getting worse, roughly what it costs, and which customers to contact today (with a draft reply).

Pure pandas + matplotlib (no web framework), so it can be tested and reused outside the app.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import triage_core as tc
import triage_engine as te

ISSUES = tc.ISSUES
LABEL = {"damaged_defective": "Damaged / defective", "wrong_item": "Wrong item", "not_as_described": "Not as described",
         "size_fit": "Fit / compatibility", "delivery_shipping": "Delivery / packaging", "customer_service": "Customer service"}
SEVERITY = te.SEVERITY

# Share of problem reviews that typically end in a refund or replacement. ASSUMPTIONS for illustration; replace with your own data.
COST_SHARE = {"damaged_defective": 0.70, "wrong_item": 0.90, "not_as_described": 0.50, "size_fit": 0.40,
              "delivery_shipping": 0.50, "customer_service": 0.25}
POSSIBLE_WEIGHT = 0.5      # an issue found only by the language model counts half in impact and cost estimates

PLAYBOOK = {
    "damaged_defective": dict(
        owner="Quality / supplier management and packaging",
        causes="Supplier defects, weak packaging, rough handling in transit.",
        actions=["Ask the supplier about the affected batch and request a quality check.",
                 "Drop-test the packaging for fragile items and add protection where it fails.",
                 "Replace or refund quickly: unresolved damage is a main driver of lost customers."]),
    "wrong_item": dict(
        owner="Warehouse / fulfilment",
        causes="Pick-and-pack errors, look-alike SKUs or variants, wrong variant shown on the listing.",
        actions=["Audit picking for the affected products (scan barcodes at packing).",
                 "Separate look-alike SKUs and variants in the warehouse.",
                 "Ship the correct item and arrange the return for the customer."]),
    "not_as_described": dict(
        owner="Merchandising / listing team",
        causes="Photos or descriptions that mislead, missing details such as size, colour or what is included.",
        actions=["Re-check the listing against the real product: photos, dimensions, colour, contents.",
                 "Add the details customers say were missing.",
                 "Offer a return or refund where the listing was misleading."]),
    "size_fit": dict(
        owner="Catalogue / vehicle-fitment data team",
        causes="Missing or wrong vehicle-fitment data (year, make, model), unclear dimensions, universal parts sold as direct fit.",
        actions=["Add or fix the vehicle fitment checker (year, make, model) on the product page.",
                 "Show exact dimensions and compatibility notes next to the add-to-cart button.",
                 "Make exchanges for the right part easy and free."]),
    "delivery_shipping": dict(
        owner="Logistics / carrier management",
        causes="Slow or unreliable carrier, long lead times, poor packaging.",
        actions=["Compare carriers and regions for the affected orders.",
                 "Give honest delivery estimates and send proactive delay notices.",
                 "Compensate late or lost orders and check packaging."]),
    "customer_service": dict(
        owner="Customer support lead",
        causes="Slow replies, unresolved cases, policy friction.",
        actions=["Review response times and unanswered tickets.",
                 "Add follow-up reminders and escalation rules.",
                 "Call or message the unhappy customers directly to recover them."]),
}

REPLY = {
    "damaged_defective": "Hi, I'm really sorry your order arrived damaged or stopped working. We'd like to put this right: we can send a replacement or refund you, whichever you prefer. Could you reply with your order number (and a photo if you have one)? Thank you for telling us.",
    "wrong_item": "Hi, I'm sorry we sent the wrong item. We'll ship the correct one right away and send a prepaid label for the return. Could you confirm your order number and delivery address?",
    "not_as_described": "Hi, thank you for the honest feedback, and I'm sorry the product didn't match the listing. We're happy to arrange a return or refund, and we're reviewing the listing so it's clearer for others. Could you share your order number?",
    "size_fit": "Hi, I'm sorry the part didn't fit your vehicle. We can arrange a free exchange for the correct part, or a refund. Could you tell us your order number and your vehicle's year, make and model?",
    "delivery_shipping": "Hi, I'm sorry about the delivery problem. We're checking where your order is right now and will update you within 24 hours. If it can't reach you soon, we'll refund you or send a replacement.",
    "customer_service": "Hi, I'm sorry you haven't had the help you needed. I'm passing this to a senior colleague who will contact you today. Could you reply with your order number and the best way to reach you?",
    "photo_damage": "Hi, thank you for your feedback and for the photo. It looks like the item may have arrived damaged. If that is right, we would like to replace it or refund you, whichever you prefer. Could you reply with your order number?",
    "photo_wrong_item": "Hi, thank you for your feedback and for the photo. It looks like the item in the photo may not be the one you ordered. If that is right, we will send the correct item and arrange the return. Could you reply with your order number?",
    None: "Hi, thank you for your feedback. We'd like to understand your experience better. Could you tell us what went wrong so we can help?",
}

GROUP_PREFERENCE = ["product", "category", "supplier"]


# ---------------------------------------------------------------- input
def standardise(df):
    """Map whatever columns the team has to a standard set. Only a review text column is required."""
    names = {str(c).lower().strip(): c for c in df.columns}
    pick = lambda *keys: next((names[k] for k in keys if k in names), None)
    rev = pick("review", "text", "comment", "body", "feedback", "review_text")
    if rev is None:
        obj = df.select_dtypes("object").columns
        rev = obj[0] if len(obj) else df.columns[0]
    out = pd.DataFrame({"review": df[rev].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()})     # blank cells (None or NaN) become empty, not the text "None"
    mapping = {"product": ("product", "product_name", "item", "sku", "title"), "category": ("category", "department"),
               "supplier": ("supplier", "vendor", "brand", "carrier"), "date": ("date", "review_date", "created", "timestamp", "time"),
               "rating": ("rating", "stars", "star_rating", "score"), "order_id": ("order_id", "order", "order_number"),
               "photo": ("photo", "image", "photo_url", "image_url", "picture")}
    for std, keys in mapping.items():
        c = pick(*keys)
        if c is not None:
            out[std] = df[c].values
    for g in GROUP_PREFERENCE:                                   # blank cells in product / category / supplier mean "unknown", not a group called ""
        if g in out:
            v = out[g].astype(object).where(out[g].notna(), "").astype(str).str.strip()
            out[g] = v.mask(v.isin(["", "nan", "None"]))
    if "date" in out:
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "rating" in out:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    ok = (out.review.str.len() > 0) & (out.review.str.lower() != "nan")
    return out[ok].reset_index(drop=True)


def group_column(q):
    return next((c for c in GROUP_PREFERENCE if c in q.columns and q[c].dropna().nunique() >= 2), None)


def analyse(d, sentiment, cfg, image_risk=None, fast=False, photo_kinds=None):
    """Triage every review and attach the team's own columns (product, date, rating ...).
    fast=True uses the keyword rules only (no language model): quick on a slow server, but it finds fewer problems.
    image_risk / photo_kinds: per-review photo evidence (None where there is no photo or it could not be scored)."""
    if fast:
        cfg = {k: {"method": "rules", "thr": 1.0} for k in ISSUES}
    q = te.triage(d.review.tolist(), sentiment, cfg, image_risk=image_risk, sort=False, photo_kinds=photo_kinds)
    q["flagged_problem_order"] = q.priority.isin(["HIGH", "MEDIUM"])      # the brief's 'automatically flag problematic orders'
    for c in d.columns:
        if c != "review":
            q[c] = d[c].values
    q["has_confirmed"] = q.confirmed != "-"
    q["has_problem"] = q.has_confirmed | (q.possible != "-")
    return q

# ---------------------------------------------------------------- numbers
def issue_table(q, aov=40.0):
    n, rows = len(q), []
    for k in ISSUES:
        conf = int(q.confirmed.str.contains(k).sum())
        poss = int(q.possible.str.contains(k).sum())
        w = conf + POSSIBLE_WEIGHT * poss
        rows.append({"issue": k, "label": LABEL[k], "confirmed": conf, "possible": poss, "reviews": conf + poss,
                     "share": (conf + poss) / n if n else 0.0, "impact": w * SEVERITY[k], "est_cost": w * COST_SHARE[k] * aov})
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


def kpis(q, aov=40.0):
    it = issue_table(q, aov)
    photos = q["photo_risk"].notna() if "photo_risk" in q.columns else pd.Series(False, index=q.index)
    return {"n": len(q), "confirmed": int(q.has_confirmed.sum()), "possible_only": int((q.has_problem & ~q.has_confirmed).sum()),
            "flagged": int(q.priority.isin(["HIGH", "MEDIUM"]).sum()), "high": int((q.priority == "HIGH").sum()),
            "medium": int((q.priority == "MEDIUM").sum()), "photos_scored": int(photos.sum()),
            "photos_flagged": int((q["photo_kind"] != "-").sum()) if "photo_kind" in q.columns else 0,
            "raised_by_photo": int((q["photo_effect"] == "photo raised the priority").sum()) if "photo_effect" in q.columns else 0,
            "money": float(it.est_cost.sum()), "aov": aov}

def issues_in(row, include_possible=False):
    out = [k.strip() for k in str(row["confirmed"]).split(",") if k.strip() and k.strip() != "-"]
    if include_possible:
        out += [k.strip() for k in str(row["possible"]).split(",") if k.strip() and k.strip() != "-"]
    return out


# Comparisons across products and over time use CONFIRMED problems only: the same keyword evidence is applied to every product and
# every week, so the comparison is consistent, and unconfirmed language-model guesses cannot create a fake hotspot or a fake trend.
def hotspots(q, col, min_n=5):
    """Which products / categories / suppliers have an unusually high share of reviews with a confirmed problem?"""
    if col is None or col not in q.columns:
        return pd.DataFrame()
    overall = q.has_confirmed.mean() if len(q) else 0.0
    rows = []
    for g, d in q.groupby(col):
        if pd.isna(g) or len(d) < min_n:
            continue
        counts = pd.Series([k for _, r in d.iterrows() for k in issues_in(r)], dtype="object").value_counts()
        rate = float(d.has_confirmed.mean())
        rows.append({col: g, "reviews": len(d), "problem_reviews": int(d.has_confirmed.sum()), "problem_rate": rate,
                     "vs_average": rate / overall if overall > 0 else np.nan,
                     "top_issue": LABEL[counts.index[0]] if len(counts) else "-"})
    h = pd.DataFrame(rows)
    return h.sort_values(["problem_rate", "reviews"], ascending=False).reset_index(drop=True) if len(h) else h


def trend(q):
    """Confirmed-problem reviews over time (weekly, or monthly for long ranges). None when there are too few dated reviews."""
    if "date" not in q.columns or q.date.notna().sum() < 12:
        return None
    d = q.dropna(subset=["date"]).copy()
    freq = "W" if (d.date.max() - d.date.min()).days <= 180 else "M"
    d["period"] = d.date.dt.to_period(freq).dt.start_time
    overall = d.groupby("period").agg(reviews=("review", "size"), problems=("has_confirmed", "sum"))
    overall["rate"] = overall.problems / overall.reviews
    per_issue = {}
    for k in ISSUES:
        has = d.confirmed.str.contains(k)
        per_issue[k] = has.groupby(d.period).sum().reindex(overall.index, fill_value=0)
    return {"overall": overall, "issues": pd.DataFrame(per_issue), "freq": freq}


def rising(tr, min_recent=3, ratio=1.5):
    """Issues whose share of reviews in the last two periods is clearly above their earlier level."""
    if tr is None or len(tr["overall"]) < 5:
        return []
    ov, iss = tr["overall"], tr["issues"]
    out = []
    for k in ISSUES:
        share = iss[k] / ov.reviews
        recent_n = int(iss[k].iloc[-2:].sum())
        recent, earlier = float(iss[k].iloc[-2:].sum() / max(ov.reviews.iloc[-2:].sum(), 1)), float(iss[k].iloc[:-2].sum() / max(ov.reviews.iloc[:-2].sum(), 1))
        if recent_n >= min_recent and recent >= ratio * max(earlier, 0.02):
            out.append({"issue": k, "label": LABEL[k], "recent": recent, "earlier": earlier, "recent_n": recent_n})
    return sorted(out, key=lambda r: -r["recent"])


# ---------------------------------------------------------------- recommendations
def recommendations(q, aov=40.0, top=3, col=None):
    it = issue_table(q, aov)
    it = it[it.confirmed >= 3].head(top)         # only recommend on issues backed by at least 3 confirmed reviews (not on language-model guesses)
    recs = []
    for rank, (_, r) in enumerate(it.iterrows(), 1):
        k, pb = r.issue, PLAYBOOK[r.issue]
        d = q[q.confirmed.str.contains(k) | q.possible.str.contains(k)]
        dc = q[q.confirmed.str.contains(k)]                     # confirmed reviews: used for the hotspot and the quotes
        hot = None
        if col and col in dc.columns:
            vc = dc[col].dropna().value_counts()
            if len(vc) and vc.iloc[0] >= 2:
                hot = {"name": vc.index[0], "n": int(vc.iloc[0]), "of": int(len(dc)),
                       "reviews_total": int((q[col] == vc.index[0]).sum())}
        quotes = [(t[:170] + ("..." if len(t) > 170 else "")) for t in dc.sort_values("priority_score", ascending=False).review.head(2)]
        recs.append({"rank": rank, "issue": k, "label": r.label, "reviews": int(r.reviews), "confirmed": int(r.confirmed), "possible": int(r.possible),
                     "share": float(r.share), "est_cost": float(r.est_cost), "owner": pb["owner"], "causes": pb["causes"],
                     "actions": pb["actions"], "hotspot": hot, "quotes": quotes})
    return recs


def reply_draft(row):
    """A draft reply for the human reviewer to edit. Uses CONFIRMED problems (or the photo finding); never an unconfirmed guess."""
    conf = [k.strip() for k in str(row["confirmed"]).split(",") if k.strip() and k.strip() != "-"]
    if conf:
        return REPLY[sorted(conf, key=lambda k: -SEVERITY[k])[0]]
    kind = row.get("photo_kind", "-") if hasattr(row, "get") else "-"
    if kind == "wrong_item":
        return REPLY["photo_wrong_item"]
    if kind == "damage":
        return REPLY["photo_damage"]
    return REPLY[None]


def cases(q):
    """Orders flagged for a person to review (HIGH and MEDIUM), with a draft reply and the reason."""
    d = q[q.priority.isin(["HIGH", "MEDIUM"])].sort_values("priority_score", ascending=False).copy()
    d["draft_reply"] = [reply_draft(r) for _, r in d.iterrows()]
    return d


def narrative(q, aov=40.0, col=None, short=False):
    """A short written brief in plain English. short=True: the headline, plus a hotspot or a worsening trend only when there is one."""
    kp, it = kpis(q, aov), issue_table(q, aov)
    if short:
        lines = [f"{kp['n']} reviews checked: {kp['flagged']} flagged for a person to review ({kp['high']} high, {kp['medium']} medium); {kp['confirmed']} name a specific problem."]
        h = hotspots(q, col)
        if len(h) and h.iloc[0].vs_average >= 1.3:
            r0 = h.iloc[0]
            lines.append(f"Hotspot: {r0[col]} ({r0.problem_rate:.0%} of {int(r0.reviews)} reviews have a confirmed problem, {r0.vs_average:.1f}x the average).")
        for r in rising(trend(q))[:1]:
            lines.append(f"Getting worse: {r['label'].lower()} ({r['recent']:.0%} of recent reviews, up from {r['earlier']:.0%}).")
        return lines
    it = it[it.reviews > 0]
    lines = [f"{kp['n']} reviews analysed. {kp['confirmed']} ({kp['confirmed'] / max(kp['n'], 1):.0%}) describe a specific operational problem"
             + (f", and {kp['possible_only']} more have a model suggestion that needs a quick human check." if kp["possible_only"] else ".")]
    if len(it):
        names = [f"{r.label} ({int(r.reviews)})" for _, r in it.head(3).iterrows()]
        lines.append("Largest problem types, weighing how many reviews and how costly each type is: " + ", ".join(names) + ".")
    lines.append(f"{kp['flagged']} orders are flagged for a person to review ({kp['high']} high priority, {kp['medium']} medium).")
    lines.append(f"Rough refund and replacement exposure: about ${kp['money']:,.0f}, assuming an average order value of ${aov:,.0f} "
                 f"(illustrative assumptions: change them to match your business).")
    h = hotspots(q, col)
    if len(h) and h.iloc[0].vs_average >= 1.3:
        r0 = h.iloc[0]
        lines.append(f"Hotspot: {r0[col]} has the highest share of reviews with a confirmed problem ({r0.problem_rate:.0%} of {int(r0.reviews)}, "
                     f"{r0.vs_average:.1f}x the average), mostly {r0.top_issue.lower()}.")
    tr = trend(q)
    for r in rising(tr)[:2]:
        lines.append(f"Getting worse: {r['label'].lower()} is confirmed in {r['recent']:.0%} of reviews in the latest two periods, up from {r['earlier']:.0%} before.")
    return lines

# ---------------------------------------------------------------- charts (light background, high contrast)
INK, MUTED, ORANGE, BLUE, RED = "#0F1A2B", "#3D4B5C", "#FF9900", "#146EB4", "#B12704"


def _fig():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8A94A0")
    ax.tick_params(colors=INK, labelsize=10)
    return fig, ax


def _empty(msg):
    fig, ax = _fig()
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=11)
    ax.axis("off")
    return fig


def plot_issues(it):
    it = it[it.reviews > 0].sort_values("impact")
    if it.empty:
        return _empty("No specific problem types found")
    fig, ax = _fig()
    ax.barh(it.label, it.confirmed, color=BLUE, label="confirmed (phrase found)")
    ax.barh(it.label, it.possible, left=it.confirmed, color="#9CC5E6", label="model suggestion, unconfirmed")
    ax.set_title("Problem types: number of reviews", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel("reviews", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    return fig


def plot_hotspots(h, col):
    if h is None or len(h) == 0:
        return _empty("Add a 'product' column to see\nwhere problems concentrate")
    h = h.head(8).iloc[::-1]
    fig, ax = _fig()
    ax.barh(h[col].astype(str).str.slice(0, 26), h.problem_rate * 100, color=ORANGE)
    ax.set_title(f"Share of reviews with a confirmed problem, by {col}", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel("% of that product's reviews", color=INK, fontsize=10)
    for i, (r, n) in enumerate(zip(h.problem_rate, h.reviews)):
        ax.text(r * 100 + 1, i, f"{r:.0%} of {int(n)}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 125)
    fig.tight_layout()
    return fig


def plot_trend(tr):
    if tr is None:
        return _empty("Add a 'date' column to see\nwhether problems are rising")
    ov = tr["overall"]
    fig, ax = _fig()
    ax.plot(ov.index, ov.rate * 100, marker="o", color=RED, linewidth=2)
    ax.set_title("Reviews with a confirmed problem over time", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel("% of reviews", color=INK, fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- Excel report
def excel_report(path, q, aov=40.0):
    col = group_column(q)
    it, hot, recs, cs = issue_table(q, aov), hotspots(q, col), recommendations(q, aov, 3, col), cases(q)
    brief = pd.DataFrame({"Summary": narrative(q, aov, col)})
    fix = pd.DataFrame([{
        "Priority": r["rank"], "Issue": r["label"], "Reviews": r["reviews"], "Confirmed": r["confirmed"], "Possible": r["possible"],
        "Est. cost (illustrative)": round(r["est_cost"]), "Owner": r["owner"], "Likely causes": r["causes"],
        "What to do": "\n".join("- " + a for a in r["actions"]),
        "Where": (f"{r['hotspot']['name']} ({r['hotspot']['n']} of {r['hotspot']['of']} confirmed reviews on this issue)" if r["hotspot"] else ""),
        "What customers say": "\n".join('"' + x + '"' for x in r["quotes"])} for r in recs])
    if len(cs):
        contact = cs[["priority", "priority_text_only", "photo_effect", "photo_kind", "confirmed", "possible", "review", "evidence", "draft_reply"]].rename(columns={
            "priority": "Final priority", "priority_text_only": "Review alone", "photo_effect": "Photo effect", "photo_kind": "Photo finding",
            "confirmed": "Confirmed problems", "possible": "Model suggestion (unconfirmed)", "review": "Customer wrote",
            "evidence": "Evidence", "draft_reply": "Draft reply (not sent)"})
        contact["Reviewer decision (confirm / dismiss)"] = ""
        contact["Reviewer notes"] = ""
    else:
        contact = pd.DataFrame()
    allr = q.drop(columns=[c for c in ("img", "photo_info") if c in q.columns]).copy()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        brief.to_excel(xw, sheet_name="Summary", index=False)
        fix.to_excel(xw, sheet_name="What to fix first", index=False)
        if len(hot):
            hot.to_excel(xw, sheet_name="Hotspots", index=False)
        contact.to_excel(xw, sheet_name="Orders to review", index=False)
        allr.to_excel(xw, sheet_name="All reviews", index=False)
        from openpyxl.styles import Alignment
        widths = {"Summary": {"A": 130}, "What to fix first": {"B": 22, "G": 30, "H": 40, "I": 60, "J": 40, "K": 60},
                  "Orders to review": {"G": 70, "H": 40, "I": 80, "J": 24, "K": 30}, "All reviews": {"A": 80}}
        for sheet, cols in widths.items():
            if sheet in xw.sheets:
                ws = xw.sheets[sheet]
                for c, w in cols.items():
                    ws.column_dimensions[c].width = w
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return path