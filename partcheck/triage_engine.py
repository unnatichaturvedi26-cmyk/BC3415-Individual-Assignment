"""Triage engine: turns (review text [+ photo risk]) into a ranked, explained, actionable work queue.

Depends on triage_core.py (issue rules, zero-shot detector, clean()).
Pipeline per review:  sentiment (calibrated P(negative))  +  issue detector  (+ image risk)
                      -> priority score -> bucket (HIGH / MEDIUM / LOW) -> evidence + suggested action
"""
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import triage_core as tc
from triage_core import ISSUES, ACTIONS

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# Business severity of each issue (0-1). These are ASSUMPTIONS about relative cost
# (refund + reship + reputation). A real retailer would replace them with its own cost data.
SEVERITY = {"damaged_defective": 0.9, "wrong_item": 0.9, "delivery_shipping": 0.7,
            "customer_service": 0.7, "not_as_described": 0.6, "size_fit": 0.4}
IMAGE_SEVERITY = 0.8          # weight of the photo evidence (damage / wrong item visible in the customer photo)
HIGH, MEDIUM = 0.65, 0.35     # bucket thresholds on the priority score


def harden_config(cfg):
    """Where zero-shot alone was chosen, add the keyword rules too (union).
    Rules are precise and can only add recall, which is what an F2-style objective rewards."""
    out = {}
    for k, v in cfg.items():
        v = dict(v)
        if v["method"] == "zs":
            v["method"] = "either"
        out[k] = v
    return out


class SentimentModel:
    """TF-IDF + Logistic Regression with isotonic calibration of the class probabilities.
    Calibrated probabilities are used as a risk signal; the hard label uses the raw argmax (better macro-F1)."""

    def __init__(self):
        self.pipe, self.cals = None, None

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.pipe = Pipeline([
            ("f", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)),
            ("c", LogisticRegression(C=1, max_iter=1000, class_weight="balanced"))]).fit(list(X_train), np.asarray(y_train))
        raw = self.pipe.predict_proba(list(X_cal))
        y_cal = np.asarray(y_cal)
        self.cals = [IsotonicRegression(out_of_bounds="clip").fit(raw[:, k], (y_cal == k).astype(int)) for k in range(3)]
        return self

    def _raw(self, texts):
        return self.pipe.predict_proba([tc.clean(t) for t in texts])

    def predict_proba(self, texts):
        raw = self._raw(texts)
        Q = np.column_stack([self.cals[k].predict(raw[:, k]) for k in range(3)]) + 1e-6
        return Q / Q.sum(axis=1, keepdims=True)

    def predict_label(self, texts):
        return self._raw(texts).argmax(1)

    def explain(self, text, top=5):
        """Words whose removal lowers confidence in the predicted class the most (occlusion)."""
        words = tc.clean(text).split()[:60]
        if not words:
            return []
        base = self._raw([" ".join(words)])[0]
        k = int(base.argmax())
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        drop = np.log(base[k] + 1e-6) - np.log(self._raw(variants)[:, k] + 1e-6)     # log-scale so saturated predictions still rank words
        order = [i for i in np.argsort(-drop)[:top] if drop[i] > 0.02]
        return [(words[i].strip(".,!?;:\"'()[]"), float(drop[i])) for i in order if words[i].strip(".,!?;:\"'()[]")]


class TransformerSentiment:
    """Fine-tuned DistilBERT sentiment model with isotonic-calibrated probabilities (same interface as SentimentModel).
    One epoch is enough on this task (validation macro-F1 did not improve over epochs 2-3 in Stage 5)."""

    def __init__(self, name="distilbert-base-uncased", max_len=128, epochs=1, bs=32, lr=3e-5, seed=42):
        self.name, self.max_len, self.epochs, self.bs, self.lr, self.seed = name, max_len, epochs, bs, lr, seed
        self.model, self.tok, self.cals, self.dev = None, None, None, None

    def _enc(self, texts, pad):
        e = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=pad, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    def fit(self, X_train, y_train, X_cal, y_cal):
        import math, torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=3).to(self.dev)
        X_train, y_np = list(X_train), np.asarray(y_train)
        ids, mask = self._enc(X_train, "max_length")
        y = torch.tensor(y_np)
        w = torch.tensor(len(y_np) / (3 * np.bincount(y_np, minlength=3)), dtype=torch.float).to(self.dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        steps = self.epochs * math.ceil(len(y_np) / self.bs)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        amp = self.dev == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp else None
        for _ in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(y_np))
            for i in range(0, len(y_np), self.bs):
                idx = perm[i:i + self.bs]
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    lg = self.model(input_ids=ids[idx].to(self.dev), attention_mask=mask[idx].to(self.dev)).logits
                loss = lossf(lg.float(), y[idx].to(self.dev))
                opt.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
        raw = self._raw(list(X_cal))
        y_cal = np.asarray(y_cal)
        self.cals = [IsotonicRegression(out_of_bounds="clip").fit(raw[:, k], (y_cal == k).astype(int)) for k in range(3)]
        return self

    def _raw(self, texts, bs=128):
        import torch
        self.model.eval()
        texts = [tc.clean(t) for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                ids, mask = self._enc(texts[i:i + bs], True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(self.dev == "cuda")):
                    lg = self.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).logits
                out.append(torch.softmax(lg.float(), -1).cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, texts):
        raw = self._raw(texts)
        Q = np.column_stack([self.cals[k].predict(raw[:, k]) for k in range(3)]) + 1e-6
        return Q / Q.sum(axis=1, keepdims=True)

    def predict_label(self, texts):
        return self._raw(texts).argmax(1)

    def explain(self, text, top=5):
        """Words whose removal lowers confidence in the predicted class the most (occlusion)."""
        words = tc.clean(text).split()[:60]
        if not words:
            return []
        base = self._raw([" ".join(words)])[0]
        k = int(base.argmax())
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        drop = np.log(base[k] + 1e-6) - np.log(self._raw(variants)[:, k] + 1e-6)     # log-scale so saturated predictions still rank words
        order = [i for i in np.argsort(-drop)[:top] if drop[i] > 0.02]
        return [(words[i].strip(".,!?;:\"'()[]"), float(drop[i])) for i in order if words[i].strip(".,!?;:\"'()[]")]


def evidence(text, flags):
    """Keyword phrases that triggered the rules (only for issues the rules actually fired on)."""
    t = tc.clean(text)
    out = {}
    for k in ISSUES:
        if flags.get(k):
            m = [x.group(0) for x in re.finditer(tc.PATTERNS[k], t, flags=re.I)]
            if m:
                out[k] = sorted(set(m))[:3]
    return out


def priority_score(p_neg, flagged, image_risk=None, severity=None):
    """0..1 urgency. Issues combine (noisy-OR: more issues -> more urgent); a negative tone scales the score
    between 60% and 100% of the issue risk, so a calm review with a real issue is still surfaced (a 'hidden problem').
    Negative tone with no identifiable issue stays low."""
    sev = severity or SEVERITY
    keep = 1.0
    for k in flagged:
        keep *= 1 - sev[k]
    if image_risk is not None:
        keep *= 1 - IMAGE_SEVERITY * float(image_risk)
    risk = 1 - keep
    if risk == 0:
        return 0.3 * float(p_neg)
    return risk * (0.6 + 0.4 * float(p_neg))


def bucket(score):
    return "HIGH" if score >= HIGH else ("MEDIUM" if score >= MEDIUM else "LOW")


PHOTO_FLAG = 0.5          # a photo whose risk is at or above this makes an order eligible for a flag, whatever the review says
_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# A happy-sounding review that still contains a real complaint usually turns on a contrast word: "looks great, BUT it doesn't fit my Civic".
# A confirmed issue phrase that comes AFTER such a word counts as a mixed complaint and is flagged (MEDIUM) even when the overall tone is positive.
# Measured on 3,000 five-star reviews: this touches 2.0% of them, against 8.8% if every confirmed phrase were flagged (most of those are negated or irrelevant mentions).
CONTRAST = re.compile(r"\b(?:but|however|unfortunately|although|though|except|only (?:problem|issue|downside|complaint)|sadly)\b")


def mixed_complaint(text, ev):
    """True if a confirmed issue phrase appears after a contrast word. ev = {issue: [matched phrases]} from evidence()."""
    t = tc.clean(text)
    pos = [m.start() for m in CONTRAST.finditer(t)]
    if not pos:
        return False
    for phrases in ev.values():
        for ph in phrases:
            i = t.find(str(ph).lower())
            if i >= 0 and any(p < i for p in pos):
                return True
    return False


CONF_CHECK = 0.3         # a confirmed problem type plus at least this much unhappiness is flagged
NEG_CHECK = 0.7          # a review this likely to be unhappy is never waved through as "no issue", even when no problem type is matched


def decide_bucket(score, irisk, n_confirmed, require_confirmation=True, p_neg=None, mixed=False):
    b = bucket(score); strong = irisk is not None and float(irisk) >= PHOTO_FLAG; needs_check = False
    if require_confirmation and b == "HIGH" and n_confirmed == 0 and not strong:
        b, needs_check = "MEDIUM", True
    if b == "LOW" and p_neg is not None and float(p_neg) >= NEG_CHECK:          # clearly unhappy, no problem type found: a person should look
        b, needs_check = "MEDIUM", True
    if b == "LOW" and n_confirmed > 0 and mixed:                                     # "looks great, but it doesn't fit"
        b = "MEDIUM"
    if b == "LOW" and n_confirmed > 0 and p_neg is not None and float(p_neg) >= CONF_CHECK:   # a named problem in a mixed or unhappy review is never "no issue"
        b = "MEDIUM"
    raised = False
    if strong and b == "LOW":
        b, raised = "MEDIUM", True
    return b, needs_check, raised


def triage(texts, sentiment, issue_config=None, flags=None, image_risk=None, severity=None, sort=True,
           require_confirmation=True, photo_kinds=None):
    """Score a batch of reviews. Pass `flags` (bool DataFrame, RangeIndex) to reuse precomputed issue flags.
    image_risk: per-review photo risk (0..1) or None (no photo / photo not scored). photo_kinds: per-review None | 'damage' | 'wrong_item'.
    An issue is CONFIRMED when a keyword rule matched a phrase; otherwise it is only POSSIBLE (found by the language model alone).
    Output keeps three things separate: the text analysis (sentiment, confirmed/possible issues, priority_text_only),
    the photo evidence (photo_kind, photo_risk) and the final decision (priority, photo_effect)."""
    texts = list(texts)
    P = sentiment.predict_proba(texts)
    lab = sentiment.predict_label(texts)
    if flags is None:
        flags = tc.detect_issues(texts, issue_config)
    flags = flags.reset_index(drop=True)
    rows = []
    for i, t in enumerate(texts):
        fl = [k for k in ISSUES if bool(flags.loc[i, k])]
        irisk = None if image_risk is None else image_risk[i]
        kind = None if photo_kinds is None else photo_kinds[i]
        ev = evidence(t, flags.loc[i].to_dict())
        confirmed = [k for k in fl if k in ev]
        possible = [k for k in fl if k not in ev]
        s_text = priority_score(P[i, 0], fl, None, severity)
        mixed = mixed_complaint(t, {k: v for k, v in ev.items() if k in confirmed})
        b_text, _, _ = decide_bucket(s_text, None, len(confirmed), require_confirmation, P[i, 0], mixed)
        s = priority_score(P[i, 0], fl, irisk, severity)
        b, needs_check, raised = decide_bucket(s, irisk, len(confirmed), require_confirmation, P[i, 0], mixed)
        if irisk is None:
            effect = "no photo evidence used"
        elif _RANK[b] > _RANK[b_text]:
            effect = "photo raised the priority"
        else:
            effect = "photo did not change the priority"
        actions = [ACTIONS[k] for k in confirmed]
        if kind == "wrong_item":
            actions.append("Photo: " + ACTIONS["wrong_item"])
        elif kind == "damage":
            actions.append("Photo: " + ACTIONS["damaged_defective"])
        if not actions:
            actions = ["Ask the customer what went wrong: no specific problem was identified."] if float(P[i, 0]) >= 0.5 else ["No action (monitor)."]
        ev_txt = "; ".join(f"{k}: {'/'.join(v)}" for k, v in ev.items())        # TEXT evidence only: the photo's finding has its own columns
        rows.append({
            "review": t, "p_negative": round(float(P[i, 0]), 2), "sentiment": SENTIMENT_LABELS[lab[i]],
            "issues": ", ".join(fl) or "-",
            "confirmed": ", ".join(confirmed) or "-", "possible": ", ".join(possible) or "-", "needs_check": needs_check,
            "evidence": ev_txt or "-",
            "priority_score": round(float(s), 3), "priority": b,
            "priority_text_only": b_text, "score_text_only": round(float(s_text), 3),
            "photo_kind": kind or "-", "photo_risk": None if irisk is None else round(float(irisk), 3), "photo_effect": effect,
            "suggested_action": " | ".join(actions)})
    df = pd.DataFrame(rows)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True) if sort else df

# ---------------- saving / loading the deployed sentiment model (used by the app) ----------------
def save_sentiment(model, path):
    import os, joblib
    os.makedirs(path, exist_ok=True)
    if isinstance(model, TransformerSentiment):
        hf = os.path.join(path, "hf_model")
        model.model.save_pretrained(hf)
        model.tok.save_pretrained(hf)
        joblib.dump({"cals": model.cals, "max_len": model.max_len, "name": model.name}, os.path.join(path, "calibration.joblib"))
        kind = "transformer"
    else:
        joblib.dump(model, os.path.join(path, "tfidf_model.joblib"))
        kind = "tfidf"
    with open(os.path.join(path, "type.txt"), "w") as f:
        f.write(kind)


def load_sentiment(path):
    import os, joblib
    kind = open(os.path.join(path, "type.txt")).read().strip()
    if kind == "tfidf":
        return joblib.load(os.path.join(path, "tfidf_model.joblib"))
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    meta = joblib.load(os.path.join(path, "calibration.joblib"))
    hf = os.path.join(path, "hf_model")
    m = TransformerSentiment(name=meta["name"], max_len=meta["max_len"])
    m.dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.tok = AutoTokenizer.from_pretrained(hf)
    m.model = AutoModelForSequenceClassification.from_pretrained(hf).to(m.dev)
    m.cals = meta["cals"]
    return m
