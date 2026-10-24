"""Automotive review data for the AutoParts Review Triage project.

Source: Amazon Reviews 2023 (McAuley Lab), category "Automotive", streamed from the Hugging Face Hub (no multi-gigabyte download).
The file is read once, top to bottom, and split by POSITION so the parts never overlap:
    lines 0 .. 300,000        -> star-balanced train / validation / test for the sentiment model
    lines 300,000 .. 420,000  -> pool for the hand-labelled issue sets (never used for training)
    lines 420,000 .. 480,000  -> pool for the business demo file (never used for training or labelling)
"""
import json
import itertools
import re

import numpy as np
import pandas as pd

PATH = "datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Automotive.jsonl"
SEED = 42
SEGMENTS = {"model": (0, 300_000), "label_pool": (300_000, 420_000), "demo_pool": (420_000, 480_000)}


def stream(start=0, stop=480_000):
    """Read review lines [start, stop) as a DataFrame with text, stars, timestamp, product id and whether the review has a photo."""
    from huggingface_hub import HfFileSystem
    rows = []
    with HfFileSystem().open(PATH, "r", encoding="utf-8") as h:
        for line in itertools.islice(h, start, stop):
            r = json.loads(line)
            rows.append({"text": (r.get("text") or "").strip(), "stars": int(r["rating"]), "ts": r.get("timestamp"),
                         "asin": r.get("parent_asin"), "has_photo": bool(r.get("images"))})
    return pd.DataFrame(rows)


def tidy(df, min_chars=15):
    """Drop empty, very short and duplicate reviews."""
    df = df[df.text.str.len() >= min_chars]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def star_balanced(df, per_star, seed=SEED):
    """per_star reviews of each star rating (1 to 5), shuffled."""
    parts = [df[df.stars == s].sample(n=per_star, random_state=seed) for s in range(1, 6)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def model_splits(df, per_star_train=8000, per_star_eval=1000, seed=SEED):
    """Star-balanced train / validation / test with no overlap between them. Stars are stored 0..4 like the original notebook."""
    df = tidy(df)
    out, rest = {}, df
    for name, n in [("test", per_star_eval), ("validation", per_star_eval), ("train", per_star_train)]:
        part = star_balanced(rest, n, seed)
        out[name] = part
        rest = rest[~rest.text.isin(set(part.text))]
    for d in out.values():
        d["label"] = d.stars - 1
    return out


# ---- product groups for the business demo: inferred from the review text (keywords), not invented
GROUPS = [("Wiper blades", r"\bwipers?\b"), ("Floor mats", r"\bfloor ?mats?\b|\bmats?\b"), ("Seat covers", r"\bseat ?covers?\b"),
          ("Brakes", r"\bbrakes?\b|\bbrake pads?\b|\brotors?\b"), ("Tires & wheels", r"\btires?\b|\btyres?\b|\bwheels?\b|\blug nuts?\b"),
          ("Lighting", r"\bled\b|\bbulbs?\b|\bheadlights?\b|\bfog lights?\b|\btail ?lights?\b"),
          ("Oil & car care", r"\boil\b|\bfluid\b|\bcoolant\b|\bwax\b|\bpolish\b|\bcleaner\b|\blubricant\b"),
          ("Phone mounts & dash cams", r"\bphone\b|\bmount\b|\bholder\b|\bdash ?cam\b"),
          ("Batteries & chargers", r"\bbatter(?:y|ies)\b|\bcharger\b|\bjump\b|\binverter\b")]


def product_group(text):
    t = str(text).lower()
    for name, pat in GROUPS:
        if re.search(pat, t):
            return name
    return "Other accessories"


def demo_file(df, n_unhappy=150, n_happy=120, seed=SEED):
    """Real review text, real stars and real dates. The product GROUP is inferred from keywords in the review.
    Enriched with unhappy reviews (1 to 3 stars) so the demo has problems to find: this is NOT a typical star mix."""
    df = tidy(df)
    df = df[df.text.str.len().between(25, 600)]
    pick = pd.concat([df[df.stars <= 3].sample(n=n_unhappy, random_state=seed), df[df.stars >= 4].sample(n=n_happy, random_state=seed)])
    pick = pick.sample(frac=1, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"review": pick.text.str.replace(r"\s+", " ", regex=True), "product": pick.text.map(product_group),
                         "date": pd.to_datetime(pick.ts, unit="ms").dt.strftime("%Y-%m-%d"), "rating": pick.stars,
                         "order_id": [f"ORD-{i + 1:04d}" for i in range(len(pick))]})
