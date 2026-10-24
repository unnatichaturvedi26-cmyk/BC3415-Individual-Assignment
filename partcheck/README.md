# PartCheck

Customer review and photo triage for an online automotive parts retailer (NTU BC3415 project).

A returns team gets reviews with text and customer photos. PartCheck:

1. **Reads the review**: sentiment, the type of problem (damaged or defective, does not fit, wrong item, not as described, delivery, customer service) and which part is named.
2. **Looks at the picture**: is it an auto part at all, does it match the part the review or the order is about, and is there a possible defect.
3. **Combines both** into a High, Medium or Low priority with the reasons (`combine()` in `triage_ui.py`).
4. **Sentiment dashboard**: for a spreadsheet of reviews, shows sentiment split, trend, product comparison, what drives unhappiness and a list of reviews to look at.

A person decides what happens next. The app never sends messages or issues refunds.

## Run it on your computer

```bash
git clone https://github.com/<your-username>/partcheck.git
cd partcheck
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On the first run the trained models (about 270 MB) are downloaded once from the Hugging Face repo `unnati002/review-triage-files`. To use another repo: set `MODEL_REPO`.
A computer without a GPU works: single orders take a few seconds, and the dashboard analyses the first 80 reviews at a time (500 with a GPU).

## Public link

A free Hugging Face Space runs the same code permanently: run `python deploy_space.py` (needs a Hugging Face write token) and share `https://huggingface.co/spaces/<your-username>/partcheck`.

## Files

| File | What it does |
|---|---|
| `app.py` | The web app (two screens) |
| `triage_ui.py` | Screen logic: picture checks, the combination rule, dashboard statistics and charts |
| `triage_engine.py` | Sentiment model, priority engine |
| `triage_core.py` | Issue rules, the automotive part list, actions |
| `triage_image.py` | Picture models (CLIP): what the picture shows, damage checks |
| `triage_insights.py` | Dashboard tables and summaries |
| `auto_data.py` | Data loading for the notebook |
| `AutoParts_Review_Triage_FINAL.ipynb` | Training, evaluation and the numbers quoted in the report |
| `colab_picture_eval.py` | Evaluates the picture checks on Colab |
| `tests/` | Logic tests (no GPU needed): `python tests/test_partcheck.py` |

## Honest limits

- Review versus picture: reliable for clearly different parts (a bottle against a tyre), weak on messy real customer photos, where it says "cannot compare".
- The validated damage check covers only 7 factory-style part types. Other pictures are "not assessed".
- A picture can never confirm that a part fits a vehicle: fitment needs the part number and vehicle data.
- Issue labels were made by an AI assistant, checked on a sample.
