# PartCheck: code review

Reviewed: the `partcheck/` folder of BC3415-Individual-Assignment (app, 5 logic modules, tests, deploy script, notebook headings).
Existing tests (59) passed before any change and still pass. 12 new regression tests and 31 screen tests were added.

**What could NOT be checked:** the trained models live on Hugging Face, which the review environment could not reach. All screen tests therefore use the project's own stand-in models. The real models, real CLIP photo checks and real timings were not run. `issue_config.json` (which decides whether the zero-shot model is used) was not visible either. Run the app once locally and click through the examples before submitting.

## How it worked

Two screens in Gradio. *Check an order*: text model (sentiment + keyword rules for six problem types + part lexicon) and CLIP zero-shot picture checks, merged by `combine()` into High/Medium/Low. *Sentiment dashboard*: batch version of the text model with charts and a review queue. The logic modules had no Gradio dependency, which made moving to Streamlit straightforward.

## Flaws found

### Fixed

| # | Problem | Why it matters | Fix |
|---|---|---|---|
| 1 | The export always wrote `partcheck_reviews.csv` in the server's working folder | On a public link, two visitors overwrite each other's file and can download each other's customer data | Download is built in memory per visitor (`csv_bytes`); `export_csv` uses a private temp file |
| 2 | Damage rule `missing (?:\\w+ ){0,2}(?:hardware...)` had an extra backslash | "missing mounting hardware" / "missing all the screws" were never flagged as defects (only the exact words "missing hardware" matched) | Regex corrected in `triage_core.py` |
| 3 | Part lexicon matched ordinary phrases | "went nuts" → *Screws, bolts & nuts*; "lightweight" → *Bulbs*; "starter kit" → *Alternator* | Idiom filter before the lexicon runs |
| 4 | Picture vs review compared only the first part named | A review naming a tyre and floor mats, with a floor-mat photo, could be called a mismatch | Match if the picture fits any named part; mismatch only if it fits none |
| 5 | CSV reading crashed on Windows-1252 or semicolon files; `.xls` needed `xlrd`, which was not in requirements; errors were cryptic | Typical Excel exports failed | Encoding fallbacks, delimiter fallback, `xlrd` added, readable messages |
| 6 | Charts built with global `pyplot` and never closed | Memory grows with every dashboard run; pyplot is not thread-safe with several visitors | Charts are plain `Figure` objects |
| 7 | Demo button said "298 real reviews" but a CPU server analysed only 80 | The demo silently showed 27% of its data | Count read from the file; CPU limit raised to 300 (`MAX_ROWS` to change) |
| 8 | Single-order screen could lazy-load a ~700 MB zero-shot model on CPU while the dashboard used rules only | The same review could get different answers on each tab; memory spike on a small host | One shared mode (`PARTCHECK_RULES_ONLY`), default rules-only without a GPU, shown in the sidebar |
| 9 | Start-up downloaded 3 example photos from Amazon (10 s timeout each) and wrote into the app folder | Slow or failing start-up offline; crash on a read-only host | Examples fetched only when clicked, cached, writable-folder fallback, never raises |
| 10 | Customer pictures were not rotated by their phone orientation flag; transparent PNGs turned black; huge photos were not shrunk | Sideways photos give wrong picture checks; wasted memory | `load_picture()` applies EXIF rotation, flattens on white, caps at 1280 px |
| 11 | Errors were swallowed (`except Exception` with no log) | Impossible to debug on a hosted app | Logged with traceback; the visitor still sees a short message |
| 12 | No protection when two visitors trigger the lazy model load at once | Race on first use | One analysis at a time (lock) |
| 13 | Model download had no revision pin and a raw traceback on failure | A change in the model repo silently changes results; offline start gives an unreadable error | `MODEL_REVISION`, readable errors, "Try again" button |
| 14 | A public link had no access control | Anyone can use your server's CPU | Optional `APP_PASSWORD` |
| 15 | `triage_insights.py` had a manager's brief, playbook (owner, causes, actions), hotspots and an Excel report with draft replies, but the app never showed them | Finished features unreachable | Shown in the dashboard ("Manager's brief") with CSV and Excel downloads. No invented money figures in the page, as the original tests required; the Excel report keeps its labelled-illustrative cost sheet |
| 16 | Customer text containing `$...$` would render as a formula in Streamlit markdown | Garbled reviews | `$` escaped everywhere; HTML in reviews is escaped and tested |
| 17 | README said the Hugging Face Space runs "permanently"; requirements were unpinned and `gradio` pulled in | Overpromise; installs differ between machines | Honest hosting notes; Gradio removed; `streamlit<2`, CPU-only PyTorch index |
| 18 | The Gradio-only theme (CSS selectors, forced-light JS) | Would not work in Streamlit | `.streamlit/config.toml` light theme + new CSS selectors |

### Still open (needs your data, Colab, or a decision)

1. **Pickled scikit-learn objects** (TF-IDF model, calibrators) only load reliably with the scikit-learn version they were saved with. Run `pip freeze | grep -i -E "scikit|joblib|torch|transformers|sentence"` in the Colab where you ran the notebook and pin those versions in `requirements.txt`. This is the most likely cause of a first-deploy failure.
2. **The models repo is a single point of failure.** If `unnati002/review-triage-files` is renamed, made private or deleted, the app stops. Keep a local copy of `artifacts/`, and set `MODEL_REVISION` to a commit hash.
3. **Picture damage check** covers 7 factory-style part types, and `T_DAMAGE` is deliberately off until `colab_picture_eval.py` shows it is reliable. Run that script and decide before the final demo.
4. **Severity weights and cost shares** (`SEVERITY`, `COST_SHARE`) are assumptions, not data. Say so in the report (the notebook already does).
5. **Issue labels were made by an AI assistant** and only sampled. Hand-check more if you quote precision/recall.
6. **Part names are keyword-based**, so other rare words can still name a part wrongly. Treat "Part named" as a hint.
7. **`triage_insights.standardise`** treats a column called `title` as the product name. In review datasets `title` is often the review headline, which makes the product chart meaningless. Rename the column before uploading, or drop `"title"` from the mapping if your data looks like that.
8. **No automated test with the real models** (CI would need the 270 MB download). Consider one smoke test you run by hand before each deploy.
9. **Ethics**: reviews and photos are personal data. The app stores nothing on disk and keeps results only in the visitor's session, but a public link plus real customer data is a decision for the business, not just the code.

## Suggestions that were not built (optional)

- Batch picture upload for the dashboard (a `photo` column is already recognised but ignored).
- A reviewer "confirm / dismiss" button that saves decisions, to measure real precision over time.
- A small `metrics.json` panel in the About tab, since the notebook already writes it.
