# PartCheck

Customer review and photo triage for an online automotive parts retailer (NTU BC3415 project). Now a **Streamlit** app: it runs on your computer and on a permanent public link.

A returns team gets reviews with text and customer photos. PartCheck:

1. **Reads the review**: sentiment, the type of problem (damaged or defective, does not fit, wrong item, not as described, delivery, customer service) and which part is named.
2. **Looks at the picture**: is it an auto part at all, does it match the part the review or the order is about, and is there a possible defect.
3. **Combines both** into a High, Medium or Low priority with the reasons (`combine()` in `triage_ui.py`).
4. **Sentiment dashboard**: for a spreadsheet of reviews, shows sentiment split, trend, product comparison, what drives unhappiness, a manager's brief (what to fix first, who owns it) and a list of reviews to look at. Downloads: CSV and a manager's Excel report with draft replies.

A person decides what happens next. The app never sends messages or issues refunds.

## Run it on your computer

```bash
git clone https://github.com/<your-username>/BC3415-Individual-Assignment.git
cd BC3415-Individual-Assignment/partcheck
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501. Use Python 3.11 or 3.12. On the first start the trained models (about 270 MB) are downloaded once from the Hugging Face repo `unnati002/review-triage-files` into `./_models`. If you already have the `artifacts/` folder from the notebook, put it next to `streamlit_app.py` and nothing is downloaded.

## A public link that stays up: Streamlit Community Cloud (free)

1. Push this folder to a **public GitHub repo** (the code only; the models are downloaded on first start, so the repo stays small).
2. Go to https://share.streamlit.io, sign in with GitHub, **Create app**, pick the repo and branch.
3. **Main file path**: `partcheck/streamlit_app.py` (or `streamlit_app.py` if the repo root is this folder). Streamlit finds `requirements.txt` next to it.
4. **Advanced settings**: choose **Python 3.11**. Optionally paste secrets (see `.streamlit/secrets.toml.example`), for example `APP_PASSWORD`.
5. **Deploy**. The first build and first model download take a few minutes. Your link looks like `https://<your-name>-partcheck.streamlit.app`. You can choose the subdomain.

What to expect from the free tier (check https://docs.streamlit.io/streamlit-community-cloud for current numbers):
- An app with no visitors for **12 hours goes to sleep**. The link keeps working: the next visitor presses "Wake up" and waits about a minute while the models load again. A free uptime pinger or just opening the link now and then keeps it awake.
- Memory is limited (roughly 1 to 2.7 GB). This is why, on a server without a GPU, PartCheck uses the keyword rules for problem types and does not load the 700 MB zero-shot model. If you still see "This app has gone over its resource limits", reboot the app from *Manage app*, and use the Hugging Face option below.
- Your repo must stay on GitHub, and the model repo on Hugging Face must stay public. If either is deleted the app stops. Keep a copy of the `artifacts/` folder.

## Other hosting options

- **Hugging Face Space:** Hugging Face now requires a PRO subscription (about $9 a month) to host Gradio or Docker Spaces, even on free CPU (creating one without PRO returns "402 Payment Required"). `deploy_space.py` and the `Dockerfile` work only with PRO.
- **Docker anywhere:** the same `Dockerfile` runs on any Docker host: `docker build -t partcheck . && docker run -p 7860:7860 partcheck`. These files have not been test-run.

## Settings (environment variables, or Streamlit secrets)

| Name | Meaning | Default |
|---|---|---|
| `APP_PASSWORD` | Ask visitors for a password. **Set this if the link is public and you do not want strangers using your server.** | none |
| `MODEL_REPO` | Hugging Face repo with the trained models | `unnati002/review-triage-files` |
| `MODEL_REVISION` | Commit or tag of that repo (pins the exact models) | latest |
| `ARTIFACTS` | Folder that already holds the models (skips the download) | `./artifacts` |
| `PARTCHECK_CACHE` | Where downloaded models are stored | `./_models` |
| `MAX_ROWS` | Most reviews analysed per dashboard run | 300 (500 with a GPU) |
| `PARTCHECK_RULES_ONLY` | `1` = problem types from keyword rules only; `0` = also the zero-shot language model | `1` without a GPU, `0` with one |
| `HF_TOKEN` | Only if the model repo is private | none |

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | The web app (two tabs) |
| `partcheck_runtime.py` | Finds or downloads the models, example pictures, safe loading of customer pictures (no web framework) |
| `triage_ui.py` | Screen logic: picture checks, the combination rule, dashboard statistics and charts |
| `triage_engine.py` | Sentiment model, priority engine |
| `triage_core.py` | Issue rules, the automotive part list, actions |
| `triage_image.py` | Picture models (CLIP): what the picture shows, damage checks |
| `triage_insights.py` | Manager's brief, playbook, hotspots, Excel report |
| `auto_data.py` | Data loading for the notebook |
| `AutoParts_Review_Triage_FINAL.ipynb` | Training, evaluation and the numbers quoted in the report |
| `colab_picture_eval.py` | Evaluates the picture checks on Colab |
| `Dockerfile`, `deploy_space.py` | Optional Hugging Face / Docker deployment |
| `.streamlit/config.toml` | Light theme, upload limit |
| `tests/` | `test_partcheck.py` (logic), `test_fixes.py` (bugs fixed in the review), `test_streamlit_app.py` (screens). No GPU, no downloads needed |
| `CODE_REVIEW.md` | What was wrong, what was changed, what is still open |

Run all tests: `python tests/test_partcheck.py && python tests/test_fixes.py && python tests/test_streamlit_app.py`

## Interface update (October 2026)

- Two tabs: Order assessment and Sentiment dashboard. The former About and limits tab was removed. The underlying checks and reports remain intact.
- Sample-order buttons are now a single scenario picker and Load example action.
- A CSV starter template is available under Accepted data and example format on the dashboard.
- A restrained, neutral design replaces the earlier orange/card-heavy theme.

## Honest limits

- Review versus picture: reliable for clearly different parts (a bottle against a tyre), weak on messy real customer photos, where it says "cannot compare".
- The validated damage check covers only 7 factory-style part types. Other pictures are "not assessed".
- A picture can never confirm that a part fits a vehicle: fitment needs the part number and vehicle data.
- Issue labels were made by an AI assistant, checked on a sample.
- Part names come from a keyword list, not a trained model.
- On a server without a GPU, problem types come from keyword rules only, so fewer problems are found than in the notebook.

## Redesigned build — local startup

On macOS, open Terminal, `cd` into the extracted `partcheck` folder and run
`bash start_mac.command`. Alternatively, after installing the dependencies,
run `python -m streamlit run streamlit_app.py`. Leave that Terminal open and
visit http://localhost:8501. The initial model download requires internet.
If port 8501 is busy, use `python -m streamlit run streamlit_app.py --server.port 8502`.
For Windows run `start_windows.bat` (requires Python in PATH).

### Visual check caveat
Recognising a steering wheel is distinct from verifying damage. The available
validated damage classifier covers a limited set of factory-style photographs.
When damage assessment is unsupported, the UI says **Part identified** and
**Not assessed — unverified** rather than displaying a generic recognition error.
This does not claim the image is defect-free. Supply the exact product ordered
to enable review/order-to-photo comparisons; some product families are not supported.

### New features
Download a standalone individual HTML order assessment. The dashboard review
queue can be searched by review text, product, order ID, part or issue, and
filtered by priority and sentiment, then exported as CSV. Priority assessments
now include a plain-language rationale. All existing batch exports remain.
