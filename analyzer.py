"""
Sentiment analysis engine.

Two interchangeable backends expose the same `predict_proba(texts)` call,
returning one {"negative", "neutral", "positive"} probability dict per text:

- "transformer": a pretrained RoBERTa model (3-class, handles neutral text)
- "tfidf":       the TF-IDF + Logistic Regression model trained in the notebook

Everything else (aspects, clause scoring, word explanations) is built on top of
`predict_proba`, so it works the same with either backend.
"""

import os
import re

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TFIDF_MODEL_PATH = "./notebooks/sentiment_model.pkl"

# Below this confidence a prediction is flagged for a human to check
REVIEW_THRESHOLD = 0.65

# Max words considered when explaining a single review (one model pass per word)
EXPLAIN_MAX_WORDS = 80

# Keywords that tie a clause to an aspect of the film
ASPECTS = {
    "acting":  ["acting", "actor", "actors", "actress", "cast", "performance",
                "performances", "chemistry", "lead", "leads", "character", "characters"],
    "story":   ["plot", "story", "storyline", "script", "writing", "written",
                "dialogue", "narrative", "twist", "premise"],
    "visuals": ["visuals", "visual", "cinematography", "effects", "cgi", "shot",
                "shots", "scenery", "photography", "beautiful", "stunning"],
    "sound":   ["sound", "soundtrack", "music", "score", "audio", "song", "songs"],
    "pacing":  ["pacing", "pace", "slow", "long", "dragged", "drags", "boring",
                "rushed", "length", "runtime", "fast"],
    "ending":  ["ending", "end", "finale", "conclusion", "climax"],
}

LABELS = ("negative", "neutral", "positive")

# Filler words hidden from explanations; negations stay because they flip meaning
STOP_WORDS = ENGLISH_STOP_WORDS - {"not", "no", "nor", "never", "nothing", "cannot"}


# ── Backends ──────────────────────────────────────────────────────────────────

class TransformerBackend:
    name = "transformer"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL,
            top_k=None,          # return all three class scores
            truncation=True,     # long reviews are cut at the model's 512 tokens
        )

    def predict_proba(self, texts):
        out = self.pipe(list(texts), batch_size=16)
        return [{d["label"].lower(): d["score"] for d in row} for row in out]


class TfidfBackend:
    name = "tfidf"

    def __init__(self):
        if not os.path.exists(TFIDF_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{TFIDF_MODEL_PATH}' not found. "
                "Please run the Jupyter notebook first to generate the model."
            )
        self.model = joblib.load(TFIDF_MODEL_PATH)

    def predict_proba(self, texts):
        classes = list(self.model.classes_)
        rows = self.model.predict_proba(list(texts))
        # Binary model: it never predicts neutral
        return [{"neutral": 0.0, **dict(zip(classes, map(float, r)))} for r in rows]


def load_backend():
    """Use the transformer unless SENTIMENT_MODEL=tfidf or it fails to load."""
    if os.getenv("SENTIMENT_MODEL", "transformer") == "tfidf":
        return TfidfBackend()
    try:
        return TransformerBackend()
    except Exception as e:  # missing torch, no network on first download, ...
        print(f"Transformer unavailable ({e}); falling back to TF-IDF model")
        return TfidfBackend()


# ── Analysis ──────────────────────────────────────────────────────────────────

def _summarize(probs):
    """Turn a probability dict into label, confidence and a -1..1 score."""
    label = max(LABELS, key=lambda k: probs.get(k, 0.0))
    return {
        "sentiment": label,
        "confidence": round(probs[label], 4),
        "score": round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4),
        "probabilities": {k: round(probs.get(k, 0.0), 4) for k in LABELS},
    }


def split_clauses(text):
    """Split a review into sentences, then on contrast words like 'but'."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    clauses = []
    for s in sentences:
        parts = re.split(r"\s*(?:\bbut\b|\bhowever\b|\balthough\b|\bthough\b|;)\s*", s, flags=re.I)
        clauses += [p.strip(" ,.") for p in parts if len(p.strip(" ,.").split()) >= 2]
    return clauses or [text]


def _aspects_in(clause):
    words = set(re.findall(r"[a-z']+", clause.lower()))
    return [a for a, keys in ASPECTS.items() if words & set(keys)]


class Analyzer:
    def __init__(self):
        self.backend = load_backend()
        print(f"Sentiment backend: {self.backend.name}")

    def analyze_many(self, texts, explain=False):
        """
        Analyze reviews in as few model calls as possible: every review and
        every clause of every review go through the model in one batch.
        """
        clause_lists = [split_clauses(t) for t in texts]
        flat = list(texts) + [c for cl in clause_lists for c in cl]
        probs = self.backend.predict_proba(flat)

        overall, clause_probs = probs[:len(texts)], iter(probs[len(texts):])
        results = []
        for text, clauses, p in zip(texts, clause_lists, overall):
            r = _summarize(p)
            r["needs_review"] = r["confidence"] < REVIEW_THRESHOLD

            scored = [{"text": c, **_summarize(next(clause_probs)), "aspects": _aspects_in(c)}
                      for c in clauses]
            r["clauses"] = scored
            r["aspects"] = self._aggregate_aspects(scored)
            if explain:
                r["positive_words"], r["negative_words"] = self.explain(text, p)
            results.append(r)
        return results

    def analyze(self, text):
        return self.analyze_many([text], explain=True)[0]

    @staticmethod
    def _aggregate_aspects(clauses):
        """Average clause scores per aspect, e.g. {"acting": 0.91, "story": -0.8}."""
        buckets = {}
        for c in clauses:
            for a in c["aspects"]:
                buckets.setdefault(a, []).append(c["score"])
        return {a: round(sum(v) / len(v), 4) for a, v in buckets.items()}

    def explain(self, text, base_probs, top_n=4):
        """
        Occlusion explanation: drop each word and see how much the
        positive-minus-negative score moves. Words whose removal lowers the
        score were pushing positive, and vice versa. Model-agnostic.
        """
        words = text.split()
        if len(words) < 2 or len(words) > EXPLAIN_MAX_WORDS:
            return [], []
        base = base_probs.get("positive", 0) - base_probs.get("negative", 0)
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        impacts = {}
        for w, p in zip(words, self.backend.predict_proba(variants)):
            key = re.sub(r"[^\w'-]", "", w).lower()
            if key and key not in STOP_WORDS:
                delta = base - (p.get("positive", 0) - p.get("negative", 0))
                impacts[key] = impacts.get(key, 0) + delta

        ranked = sorted(impacts.items(), key=lambda kv: kv[1])
        positive = [w for w, d in reversed(ranked) if d > 0.02][:top_n]
        negative = [w for w, d in ranked if d < -0.02][:top_n]
        return positive, negative
