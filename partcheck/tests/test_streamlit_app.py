"""Screen tests for the Streamlit app (no real models, no network): the app is driven with Streamlit's AppTest and the same stand-in models
that tests/test_partcheck.py uses.     Run:  python tests/test_streamlit_app.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("PARTCHECK_CACHE", os.path.join(ROOT, "_test_cache"))
os.environ.pop("APP_PASSWORD", None)

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

# stand-in models: run only the class definitions of test_partcheck.py (everything before its first check)
src = open(os.path.join(HERE, "test_partcheck.py"), encoding="utf-8-sig").read().split('print("== 1. TEXT ONLY ==")')[0]
ns = {"__name__": "stubs", "__file__": os.path.join(HERE, "test_partcheck.py")}
exec(compile(src, "test_partcheck_stubs", "exec"), ns)
STUB = ns["M"]

import partcheck_runtime as rt
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


def new_app():
    rt.load_models = lambda: STUB
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=120)
    return at.run()


def text(at):
    return " ".join(m.value for m in at.markdown)


print("== A. START-UP ==")
at = new_app()
check("the app starts without an exception", not at.exception, str(at.exception))
check("two tabs", len(at.tabs) == 2)
check("review box, product box and the Analyse button exist", at.text_area(key="review") is not None and at.text_input(key="ordered") is not None and at.button(key="go_order") is not None)
check("the demo button shows the real row count, not a hard-coded number", any("Demo: 298" in b.label for b in at.button))

print("== B. CHECK AN ORDER ==")
at.text_area(key="review").set_value("The bottle of motor oil arrived leaking and the box is soaked, very disappointed.").run()
at.button(key="go_order").click().run()
check("a damaged-oil review is flagged", not at.exception and "Review recommended" in text(at), str(at.exception))
at.text_area(key="review").set_value("Great product, works perfectly, thank you!").run()
at.button(key="go_order").click().run()
check("a happy review is not flagged", "No issue detected" in text(at))
at.button(key="clear_order").click().run()
check("Clear empties the form and the result", at.text_area(key="review").value == "" and "The assessment appears here." in text(at))
at.button(key="go_order").click().run()
check("pressing Analyse on an empty form shows the hint instead of crashing", not at.exception and "Add a review, a picture, or both" in text(at))
at.selectbox(key="example_choice").set_value(5).run()
at.button(key="load_example").click().run()
check("an example fills the review", at.text_area(key="review").value.startswith("Great product, thank you"))
check("the locally drawn example picture is shown", len(at.get("imgs")) >= 1 or len(at.get("image")) >= 1)
at.button(key="go_order").click().run()
check("an example with a picture produces a three-part card", not at.exception and "3. Combined decision" in text(at), str(at.exception))
at.text_area(key="review").set_value("Cost me $5 and $10 to ship and it broke").run()
at.button(key="go_order").click().run()
check("dollar signs in a review do not break the card", not at.exception and "Review recommended" in text(at))

print("== C. DASHBOARD ==")
at = new_app()
at.button(key="go_demo").click().run()
check("the demo dashboard analyses every demo review", not at.exception and "298 reviews analysed" in text(at), str(at.exception) + text(at)[:200])
check("the manager's brief is shown", "Manager's brief" in text(at))
check("KPI cards and the review list are shown", "pc-kpi" in text(at) and "pc-row" in text(at))
check("report downloads and CSV template are offered", len(at.get("download_button")) == 3, str(len(at.get("download_button"))))
at.radio(key="queue_show").set_value("Everything").run()
check("switching to Everything keeps the results", not at.exception and "pc-row" in text(at))
at = new_app()
at.button(key="go_dash").click().run()
check("nothing to analyse: a clear message", "Upload a spreadsheet, paste some reviews" in text(at))
at.text_area(key="pasted").set_value("It broke after two days, total waste of $20\nLove it, great value").run()
at.button(key="go_dash").click().run()
check("pasted reviews work", not at.exception and "2 reviews analysed" in text(at), str(at.exception))
at.text_area(key="pasted").set_value("<script>alert(1)</script> <b>bold</b> it broke, wasted $20 and $30").run()
at.button(key="go_dash").click().run()
check("HTML typed into a review is shown as text, never run", "<script>" not in text(at) and "&lt;script&gt;" in text(at) and "<b>bold</b>" not in text(at))
check("a dollar amount in a review is escaped (no LaTeX)", "$20" not in text(at) and "&#36;20" in text(at))
at.text_area(key="pasted").set_value("It broke after two days, total waste of $20\nLove it, great value").run()

print("== D. PASSWORD ==")
os.environ["APP_PASSWORD"] = "s3cret"
at = new_app()
check("with APP_PASSWORD set, the tabs are hidden until the password is entered", len(at.tabs) == 0 and at.text_input(key="pw_input") is not None)
at.text_input(key="pw_input").set_value("wrong").run()
check("a wrong password is refused", len(at.tabs) == 0 and len(at.error) >= 1)
at.text_input(key="pw_input").set_value("s3cret").run()
check("the right password opens the app", len(at.tabs) == 3, str(at.exception))
os.environ.pop("APP_PASSWORD")

print("== E. MODEL LOADING FAILURES ==")
def boom():
    raise rt.ModelsUnavailable("The trained models could not be downloaded (test).")
rt.load_models = boom
st.cache_resource.clear()
at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=60).run()
check("a model download failure shows a readable error, not a traceback", not at.exception and any("could not be downloaded" in e.value for e in at.error))

print("== F. HELPERS ==")
from PIL import Image
buf = io.BytesIO(); Image.new("RGBA", (40, 20), (255, 0, 0, 0)).save(buf, format="PNG")
im = rt.load_picture(buf.getvalue())
check("a transparent PNG becomes RGB on white, not black", im.mode == "RGB" and im.getpixel((5, 5)) == (255, 255, 255))
buf = io.BytesIO(); Image.new("RGB", (4000, 2000), (10, 10, 10)).save(buf, format="JPEG")
check("a huge picture is shrunk", max(rt.load_picture(buf.getvalue()).size) <= rt.MAX_IMAGE_SIDE)
try:
    rt.load_picture(b"not a picture"); ok = False
except ValueError:
    ok = True
check("a file that is not a picture gives a readable error", ok)
check("the CSV export is in memory and keeps accents", "é".encode("utf-8") in ui.csv_bytes(pd.DataFrame({"review": ["café"], "sentiment": ["positive"]})))
p1, p2 = ui.export_csv(pd.DataFrame({"review": ["a"]})), ui.export_csv(pd.DataFrame({"review": ["b"]}))
check("export_csv never reuses a file name (two visitors cannot overwrite each other)", p1 != p2); os.remove(p1); os.remove(p2)

os.environ["PARTCHECK_RULES_ONLY"] = "1"; a = rt.rules_only()
os.environ["PARTCHECK_RULES_ONLY"] = "0"; b = rt.rules_only(); os.environ.pop("PARTCHECK_RULES_ONLY")
check("PARTCHECK_RULES_ONLY switches the issue detector mode", a is True and b is False)

print(f"\n{PASSES} passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
