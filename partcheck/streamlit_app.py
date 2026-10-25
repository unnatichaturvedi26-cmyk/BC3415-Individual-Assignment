"""PartCheck: Streamlit web app (screen wiring only; all logic and HTML live in triage_ui.py, which is tested without any web framework).

Tab 1  Check an order       : a customer's review and/or one picture -> issue? random picture? does the picture match the review? -> recommendation
Tab 2  Sentiment dashboard  : many reviews -> how customers feel, what drives the unhappy ones, which reviews to look at, a manager's brief

Run locally:   streamlit run streamlit_app.py
Optional settings (environment variables or .streamlit/secrets.toml): APP_PASSWORD, MODEL_REPO, MODEL_REVISION, MAX_ROWS, HF_TOKEN  (see partcheck_runtime.py)
"""
import hmac
import io
import html
from datetime import datetime
import logging
import os
import re
import threading

import streamlit as st

import partcheck_runtime as rt
import triage_insights as ins
import triage_ui as ui

log = logging.getLogger("partcheck")

st.set_page_config(page_title="PartCheck | Review intelligence", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>" + ui.CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- small helpers
def md(html):
    """Shows trusted HTML built by triage_ui (every customer string in it is already escaped). `$` would start a LaTeX formula in Streamlit, and a blank line would end the HTML block."""
    st.markdown(re.sub(r"\n\s*\n", "\n", html).replace("$", "&#36;"), unsafe_allow_html=True)


def plain(text):
    """Customer text shown as markdown: no formulas."""
    return str(text).replace("$", "\\$")


def setting(name, default=""):
    """Environment variable first, then Streamlit secrets (which raise if no secrets file exists)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def show_fig(fig):
    if fig is not None:
        st.pyplot(fig)


# ---------------------------------------------------------------- optional password (a public link would otherwise let anyone use your server)
def password_gate():
    pw = setting("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    entered = st.text_input("Password", type="password", key="pw_input")
    if entered:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


# ---------------------------------------------------------------- models (loaded once per server, shared by all visitors)
@st.cache_resource(show_spinner=False)
def get_models():
    return rt.load_models()


@st.cache_resource(show_spinner=False)
def get_lock():
    return threading.Lock()         # one analysis at a time: the models are shared, and the zero-shot model loads lazily on first use


@st.cache_data(show_spinner=False)
def example_path(name):
    return rt.example_picture(name)


HERO = "<div class='pc-hero'><div class='pc-eyebrow'>AUTOMOTIVE CUSTOMER INTELLIGENCE</div><h1>PartCheck.</h1><p class='sub'>Better decisions, one review at a time.</p></div>"
md(HERO)
password_gate()

try:
    with st.spinner("Loading the models. The very first start downloads about 270 MB and can take a few minutes..."):
        MODELS = get_models()
except rt.ModelsUnavailable as e:
    st.error(str(e))
    if st.button("Try again"):
        st.rerun()
    st.stop()
except Exception as e:
    log.exception("model loading failed")
    st.error(f"The models could not be loaded ({type(e).__name__}). See the server log for details, then press Try again.")
    if st.button("Try again"):
        st.rerun()
    st.stop()
LOCK = get_lock()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Workspace")
    st.caption("Review and photo triage. All actions are recommendations for human review.")
    st.markdown("**System**")
    st.caption(f"Sentiment model: {type(MODELS.sentiment).__name__}")
    st.caption("Picture checks: " + ("on" if MODELS.rec is not None else "OFF (picture model not found)"))
    st.caption(f"Dashboard limit: {rt.max_rows()} reviews per run" + ("" if rt.has_gpu() else " (CPU server)"))
    st.caption("Problem types: " + ("keyword rules only" if rt.rules_only() else "keyword rules + language model"))

# ================================================================ tab 1
EXAMPLES = [("Does not fit my F-150", "It doesn't fit my 2018 F-150. The bolt holes are off and the listing said it would fit.", None, ""),
            ("Looks great, but no fit", "Looks great and feels well made, but it doesn't fit my 2015 Civic.", None, ""),
            ("Leaking oil, tyre photo", "The bottle of motor oil arrived leaking and the box is soaked.", "tyre", ""),
            ("Ordered oil, tyre photo", "This is not what I expected, it does not look right.", "tyre", "Valvoline synthetic motor oil 5 quart"),
            ("Floor mats, good photo", "These floor mats are great, perfect fit and easy to clean.", "floor_mat", ""),
            ("Happy review, random photo", "Great product, thank you!", "random_screenshot", "")]

for key, default in (("review", ""), ("ordered", ""), ("ex_pic", None), ("up_n", 0), ("order_html", None), ("order_result", None), ("ex_note", ""), ("dash", None)):
    st.session_state.setdefault(key, default)


def load_example(i):
    _, text, pic, ordered = EXAMPLES[i]
    st.session_state.review, st.session_state.ordered = text, ordered
    st.session_state.up_n += 1                       # a new key empties the file uploader
    st.session_state.order_html, st.session_state.order_result, st.session_state.ex_note = None, None, ""
    st.session_state.ex_pic = None
    if pic:
        path = example_path(pic)
        if path:
            st.session_state.ex_pic = path
        else:
            st.session_state.ex_note = "The example picture could not be downloaded (no internet?). Only the text was filled in; upload your own picture to try the picture checks."


def clear_order():
    st.session_state.review, st.session_state.ordered, st.session_state.ex_pic = "", "", None
    st.session_state.up_n += 1
    st.session_state.order_html, st.session_state.order_result, st.session_state.ex_note = None, None, ""


tab_order, tab_dash = st.tabs(["Order assessment", "Sentiment dashboard"])

with tab_order:
    left, right = st.columns(2, gap="large")
    with left:
        md("<div class='pc-h2'>Customer review and picture</div><p class='pc-sub'>Add a review, a picture, or both. The product ordered is optional.</p>")
        st.text_area("Customer review", key="review", height=120, placeholder="Paste or type the review")
        upload = st.file_uploader("Customer picture (optional)", type=["jpg", "jpeg", "png", "webp"], key=f"upload_{st.session_state.up_n}")
        st.text_input("Product ordered (optional)", key="ordered", placeholder="For example: Bosch wiper blades 24 inch")
        picture, picture_error = None, ""
        if upload is not None:
            try:
                picture = rt.load_picture(upload)
            except ValueError as e:
                picture_error = str(e)
        elif st.session_state.ex_pic:
            with open(st.session_state.ex_pic, "rb") as fh:
                picture = rt.load_picture(fh.read())
        if picture_error:
            st.error(picture_error)
        if picture is not None:
            st.image(picture, width=210, caption="Picture that will be checked")
        if st.session_state.ex_note:
            st.warning(st.session_state.ex_note)
        b1, b2 = st.columns([3, 1])
        go = b1.button("Analyse order", type="primary", key="go_order")
        b2.button("Clear", key="clear_order", on_click=clear_order)
        with st.container(border=True):
            st.markdown("**Explore with sample orders**")
            st.caption("Select a scenario, then load it into the form above. Nothing is analysed until you click Analyse order.")
            example_col, load_col = st.columns([3, 1], vertical_alignment="bottom")
            selected = example_col.selectbox("Example scenario", range(len(EXAMPLES)), format_func=lambda i: EXAMPLES[i][0], key="example_choice", label_visibility="collapsed")
            load_col.button("Load example", key="load_example", on_click=load_example, args=(selected,), use_container_width=True)
    with right:
        md("<div class='pc-h2'>Assessment</div><p class='pc-sub'>Review signals and visual checks are assessed independently, then combined into a suggested priority.</p>")
        slot = st.empty()
        if go:
            if picture_error:
                st.session_state.order_result = None
                st.session_state.order_html = ui.banner("err", picture_error)
            else:
                try:
                    with st.spinner("Checking the order..."), LOCK:
                        res = ui.assess_order(st.session_state.review, picture, MODELS, st.session_state.ordered)
                    st.session_state.order_html = ui.render_order(res)
                    st.session_state.order_result = res
                except Exception as e:
                    log.exception("order check failed")
                    st.session_state.order_result = None
                    st.session_state.order_html = ui.banner("err", f"Something went wrong while checking this order ({type(e).__name__}). Try again, or use a different picture.")
        with slot.container():
            md(st.session_state.order_html or "<div class='pc-empty'>Your order assessment will appear here.</div>")
        if st.session_state.order_result and not st.session_state.order_result.get("error"):
            result = st.session_state.order_result
            # Self-contained HTML report: preserves all explanations; no extra PDF dependencies.
            report_body = ui.render_order(result)
            report_html = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                           "<title>PartCheck order assessment</title><style>" + ui.CSS +
                           "body{max-width:950px;margin:35px auto;padding:20px;font-family:Arial,sans-serif}</style></head>"
                           "<body><h1>PartCheck — Order assessment</h1><p>Generated " +
                           html.escape(datetime.now().strftime('%d %B %Y')) + "</p>" + report_body + "</body></html>")
            st.download_button("Download assessment (HTML)", report_html.encode('utf-8'),
                               file_name='partcheck_order_assessment.html', mime='text/html',
                               key='dl_individual', use_container_width=True)


# ================================================================ tab 2
def build_result(df, note):
    """Runs the analysis and prepares every download, so a page refresh or a click on 'Show' does not repeat the work."""
    with LOCK:
        res = ui.analyse_reviews(df, MODELS, fast=rt.rules_only(), max_rows=rt.max_rows())
    res["note"] = note
    res["csv"] = ui.csv_bytes(res["q"])
    try:
        buf = io.BytesIO()
        ins.excel_report(buf, res["q"])
        res["xlsx"] = buf.getvalue()
    except Exception:
        log.exception("excel report failed")
        res["xlsx"] = None
    return res


@st.cache_data(show_spinner=False, max_entries=1)
def demo_result(max_rows):
    """The demo file never changes, so its analysis is computed once per server start (max_rows is only part of the cache key)."""
    path = rt.find_file("demo_auto_reviews.csv")
    if path is None:
        raise ui.InputError("The demo file demo_auto_reviews.csv was not found next to the app.")
    df, note = ui.read_table("", None, path)
    return build_result(df, note)


def show_dashboard(res):
    q = res["q"]
    notes = " ".join(([res["note"]] if res.get("note") else []) + res["notes"])
    md(ui.banner("warn" if res["notes"] else "ok", f"Done: {len(q)} reviews analysed. {notes}".strip()))
    md(ui.render_kpis(q))
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ui.fig_split(q))
    with c2:
        show_fig(ui.fig_trend(q))
    show_fig(ui.fig_products(q))
    show_fig(ui.fig_drivers(q))

    # --- manager's brief: already built in triage_insights, now visible in the app (no invented money figures here; the Excel report has them, labelled illustrative)
    try:
        col = ins.group_column(q)
        lines = [ln for ln in ins.narrative(q, col=col) if "exposure" not in ln]
        recs = ins.recommendations(q, col=col)
        hot = ins.hotspots(q, col)
    except Exception:
        log.exception("manager brief failed")
        lines, recs, hot, col = [], [], None, None
    if lines or recs:
        md("<div class='pc-h2' style='margin-top:10px'>Manager's brief</div>")
        for ln in lines:
            st.markdown("- " + plain(ln))
        for r in recs:
            with st.expander(f"{r['rank']}. {r['label']}: {r['reviews']} reviews ({r['confirmed']} confirmed by a phrase)"):
                st.markdown(f"**Owner:** {plain(r['owner'])}  \n**Likely causes:** {plain(r['causes'])}")
                st.markdown("**What to do**\n" + "\n".join("- " + plain(a) for a in r["actions"]))
                if r["hotspot"]:
                    h = r["hotspot"]
                    st.markdown(f"**Where:** {plain(h['name'])} ({h['n']} of {h['of']} confirmed reviews on this issue)")
                for quote in r["quotes"]:
                    st.markdown("> " + plain(quote))
        if hot is not None and len(hot):
            with st.expander(f"Hotspots by {col}"):
                st.dataframe(hot.rename(columns={"problem_rate": "share with a confirmed problem"}), hide_index=True)

    md("<div class='pc-h2' style='margin-top:10px'>Reviews to look at</div>")
    s1, s2 = st.columns([2, 1])
    show = s1.radio("Show", ["Needs review", "Everything"], horizontal=True, key="queue_show")
    limit = s2.slider("Most reviews listed", 10, 100, 40, step=10, key="queue_limit")
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search review or product", key="queue_query", placeholder="Search customer text, product or order ID")
    priority = f2.selectbox("Priority", ["All", "HIGH", "MEDIUM", "LOW"], key="queue_priority")
    sentiment = f3.selectbox("Sentiment", ["All", "negative", "neutral", "positive"], key="queue_sentiment")
    filtered = q
    if query.strip():
        columns = [c for c in ("review", "product", "order_id", "part", "confirmed") if c in filtered.columns]
        mask = filtered[columns].fillna("").astype(str).apply(
            lambda col: col.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
        filtered = filtered[mask]
    if priority != "All":
        filtered = filtered[filtered.priority == priority]
    if sentiment != "All":
        filtered = filtered[filtered.sentiment == sentiment]
    st.caption(f"{len(filtered):,} of {len(q):,} reviews match your filters")
    md(ui.render_queue(filtered, show, limit))
    st.download_button("Download filtered reviews (CSV)", ui.csv_bytes(filtered),
                       file_name="partcheck_filtered_reviews.csv", mime="text/csv", key="dl_filtered")

    d1, d2, _ = st.columns([1, 1, 1])
    d1.download_button("Download all reviews (CSV)", res["csv"], file_name="partcheck_reviews.csv", mime="text/csv", key="dl_csv")
    if res.get("xlsx"):
        d2.download_button("Manager report (Excel)", res["xlsx"], file_name="partcheck_report.xlsx", key="dl_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


with tab_dash:
    md("<div class='pc-h2'>Customer sentiment</div><p class='pc-sub'>Analyse a review dataset to understand sentiment, recurring issues and where to act first.</p>")
    with st.expander("Accepted data and example format"):
        st.write("Upload CSV or Excel with a review-text column. Product, rating and date columns are optional and enable more detailed breakdowns. You can also paste one review per line.")
        st.download_button("Download CSV starter template", data="review,rating,product,date\n\"The part does not fit\",2,Wiper blades,2026-10-01\n", file_name="partcheck_template.csv", mime="text/csv", key="dl_template")
    k1, k2 = st.columns(2, gap="large")
    sheet = k1.file_uploader("Reviews (Excel or CSV)", type=["csv", "xlsx", "xls"], key="sheet")
    pasted = k2.text_area("...or paste reviews, one per line", height=130, key="pasted")
    n_demo = rt.demo_rows()
    a1, a2, _ = st.columns([1, 1, 2])
    go_dash = a1.button("Analyse reviews", type="primary", key="go_dash")
    go_demo = a2.button(f"Demo: {n_demo} real reviews" if n_demo else "Demo", key="go_demo", disabled=not n_demo)

    if go_dash or go_demo:
        try:
            with st.spinner("Analysing the reviews. This takes up to a minute on a small server..."):
                if go_demo:
                    st.session_state.dash = demo_result(rt.max_rows())
                else:
                    df, note = ui.read_table(pasted, sheet, None)
                    st.session_state.dash = build_result(df, note)
        except (ValueError, ui.InputError) as e:
            st.session_state.dash = None
            md(ui.banner("err", f"Could not analyse this input. {e}"))
        except Exception as e:
            log.exception("dashboard failed")
            st.session_state.dash = None
            md(ui.banner("err", f"Something went wrong while analysing the reviews ({type(e).__name__}). Try a smaller file."))
    if st.session_state.dash is not None:
        show_dashboard(st.session_state.dash)

