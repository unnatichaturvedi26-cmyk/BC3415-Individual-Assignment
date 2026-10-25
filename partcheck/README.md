# PartCheck

Review triage for an online automotive parts retailer. NTU BC3415 Individual Assignment.

**Live app:** https://bc3415-individual-assignment-cjzx9a85mh4nfpayjuoi8i.streamlit.app

A returns team receives customer reviews. PartCheck reads each review and suggests what to check first:

1. **Sentiment:** positive, neutral or negative, with "Mixed" for a neutral review that leans negative.
2. **Problem type:** damaged or defective, does not fit, wrong item, not as described, delivery, or customer service. The evidence phrase from the review is shown.
3. **Part named:** the part the review is about, from an automotive part list.
4. **Priority:** High, Medium or Low, with the reasons. A person makes the final decision. The app does not send messages or issue refunds.
5. **Sentiment dashboard:** for a spreadsheet of reviews, shows sentiment, trend, product comparison, what drives unhappiness, a manager's brief, and a review queue. Results can be exported as CSV or as an Excel report with draft replies for a reviewer to edit.

An optional picture check is also included. It is not part of the core assignment and is not fully validated.

## Results

The text model is a fine-tuned DistilBERT, adapted to automotive reviews. It was compared with three baselines on the same held-out test set of 4,996 reviews (after removing empty and duplicate reviews). Scores are macro-F1, which gives each sentiment class equal weight.

| Model | Accuracy | Macro-F1 | 95% CI | Neutral F1 |
|---|---|---|---|---|
| Majority class | 0.40 | 0.19 | 0.186 to 0.195 | 0.00 |
| TF-IDF + Naive Bayes | 0.70 | 0.52 | 0.514 to 0.529 | 0.00 |
| TF-IDF + Logistic regression | 0.70 | 0.66 | 0.649 to 0.673 | 0.44 |
| **Fine-tuned DistilBERT (deployed)** | **0.72** | **0.69** | **0.679 to 0.703** | **0.47** |

Neutral reviews are the weakest class, which is why the app shows "Mixed" for neutral reviews that lean negative.

Problem detection was checked on 100 held-out reviews: precision and recall were both 0.84. Only three problem types were checked against labelled reviews (damaged or defective, not as described, and fit). The other three (wrong item, delivery, and customer service) use keyword rules only.

**Data:** Amazon Reviews 2023, Automotive category. Training used 40,000 reviews. The notebook documents the cleaning steps and the evaluation.

**Where the work is:** `AutoParts_Review_Triage_FINAL.ipynb` contains the data preparation, model comparison, locked test-set evaluation and problem detector. Its outputs are saved in the file.

## Run it on your computer

```bash
git clone https://github.com/unnatichaturvedi26-cmyk/BC3415-Individual-Assignment.git
cd BC3415-Individual-Assignment/partcheck
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501. Use Python 3.11 or 3.12.

On first start the trained models (about 270 MB) download once from the Hugging Face repo `unnati002/review-triage-files` into `./_models`. This needs an internet connection. If you already have an `artifacts/` folder, put it next to `streamlit_app.py` and nothing is downloaded.

On Windows you can also run `start_windows.bat`. On macOS, run `bash start_mac.command`.

## Run the tests

```bash
python tests/test_partcheck.py
python tests/test_fixes.py
python tests/test_streamlit_app.py
```

The tests need no GPU and no downloads. They use stand-in models.

## Settings

Set these as environment variables, or in Streamlit secrets (see `.streamlit/secrets.toml.example`).

| Name | Meaning | Default |
|---|---|---|
| `APP_PASSWORD` | Asks visitors for a password. Set this if the link is public. | none |
| `MODEL_REPO` | Hugging Face repo with the trained models | `unnati002/review-triage-files` |
| `MODEL_REVISION` | Commit or tag of that repo, to pin the exact models | latest |
| `ARTIFACTS` | Folder that already holds the models (skips the download) | `./artifacts` |
| `PARTCHECK_CACHE` | Where downloaded models are stored | `./_models` |
| `MAX_ROWS` | Most reviews analysed per dashboard run | 300 (500 with a GPU) |
| `PARTCHECK_RULES_ONLY` | `1` = problem types from keyword rules only, `0` = also use the zero-shot language model | `1` without a GPU |
| `HF_TOKEN` | Only needed if the model repo is private | none |

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | The web app: Order assessment and Sentiment dashboard tabs |
| `triage_ui.py` | Screen logic: the priority rule, dashboard statistics and charts |
| `triage_engine.py` | Sentiment model and priority engine |
| `triage_core.py` | Problem rules, the automotive part list and suggested actions |
| `triage_insights.py` | Manager's brief, playbook, and Excel report |
| `triage_image.py` | Optional picture checks (CLIP) |
| `partcheck_runtime.py` | Model download and loading, example pictures, safe picture loading |
| `auto_data.py` | Data loading for the notebook |
| `AutoParts_Review_Triage_FINAL.ipynb` | Training, evaluation and the numbers in this README |
| `colab_picture_eval.py` | Evaluates the optional picture checks on Colab |
| `tests/` | Logic, bug-fix and screen tests |
| `CODE_REVIEW.md` | What was checked, what was fixed, and what is still open |

## Limits

- The problem labels were generated by an AI assistant from written definitions. A sample check is planned, not complete.
- Three of the six problem types use keyword rules only and were not checked against labelled reviews.
- Neutral reviews are the hardest to classify (F1 0.47).
- Part names come from a keyword list, not a trained model.
- The app cannot confirm that a part fits a vehicle. Fitment needs the part number and vehicle data.
- The free hosting sleeps after 12 hours without visitors. The first visit after that takes about a minute to wake up.