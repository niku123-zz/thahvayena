from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import csv
import io

import db
from analyzer import ASPECTS, Analyzer

# Upper bound on reviews per batch / CSV upload (keeps CPU inference bounded)
MAX_BATCH = 500

# Column names we look for in uploaded CSVs, in order of preference
TEXT_COLUMNS = ("review", "text", "comment", "feedback", "content", "body")

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CineFeels Audience Feedback API",
    description="Sentiment, aspect analysis and triage for movie reviews.",
    version="2.0.0"
)

# Allow requests from the frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and prepare the history database on startup
analyzer = Analyzer()
db.init()

# ── Request / Response Schemas ────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    """The request body for the /predict endpoint."""
    text: str = Field(max_length=5000)
    save: bool = True    # store the result in history

class BatchRequest(BaseModel):
    """A list of reviews to analyze together."""
    reviews: list[str] = Field(max_length=MAX_BATCH)
    save: bool = True

class Clause(BaseModel):
    text: str
    sentiment: str
    confidence: float
    score: float
    aspects: list[str]

class Prediction(BaseModel):
    text: str
    sentiment: str                 # "positive", "neutral" or "negative"
    confidence: float              # probability of the predicted class
    score: float                   # positive minus negative, -1..1
    probabilities: dict[str, float]
    needs_review: bool             # confidence too low to trust
    aspects: dict[str, float]      # aspect -> average score, e.g. {"acting": 0.9}
    clauses: list[Clause]          # per-clause breakdown
    positive_words: list[str] = [] # words pushing toward positive (single only)
    negative_words: list[str] = [] # words pushing toward negative (single only)

class BatchResponse(BaseModel):
    """Summary of a batch plus every review, most negative first."""
    total: int
    counts: dict[str, int]
    needs_review: int
    aspects: dict[str, float]
    results: list[Prediction]

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_batch(texts: list[str], source: str, save: bool) -> BatchResponse:
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        raise HTTPException(400, "No reviews to analyze.")
    if len(texts) > MAX_BATCH:
        raise HTTPException(413, f"At most {MAX_BATCH} reviews per request.")

    results = [Prediction(text=t, **r) for t, r in zip(texts, analyzer.analyze_many(texts))]
    if save:
        db.save([r.model_dump() for r in results], source)

    # Most negative first, so problems surface at the top
    results.sort(key=lambda r: r.score)

    aspect_scores: dict[str, list[float]] = {}
    for r in results:
        for a, s in r.aspects.items():
            aspect_scores.setdefault(a, []).append(s)

    return BatchResponse(
        total=len(results),
        counts={k: sum(r.sentiment == k for r in results)
                for k in ("positive", "neutral", "negative")},
        needs_review=sum(r.needs_review for r in results),
        aspects={a: round(sum(v) / len(v), 4) for a, v in aspect_scores.items()},
        results=results,
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check — returns OK and which model backend is active."""
    return {"status": "ok", "backend": analyzer.backend.name}


@app.post("/predict", response_model=Prediction)
def predict_sentiment(request: ReviewRequest):
    """
    Analyze one review: overall sentiment, per-aspect scores, a clause
    breakdown and the words that drove the decision.

    Example request body:
        {"text": "Great visuals but the plot was awful."}
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Please enter some text.")

    result = Prediction(text=text, **analyzer.analyze(text))
    if request.save:
        db.save([result.model_dump()], "single")
    return result


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Analyze many reviews at once (e.g. a day's worth of audience feedback)."""
    return run_batch(request.reviews, "batch", request.save)


@app.post("/upload-csv", response_model=BatchResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Analyze a CSV of reviews. Uses the first column named like
    review/text/comment/feedback, otherwise the first column.
    """
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise HTTPException(400, "The CSV file is empty.")

    header = [h.strip().lower() for h in rows[0]]
    col = next((header.index(c) for c in TEXT_COLUMNS if c in header), None)
    if col is None:
        col, body = 0, rows          # no recognizable header: treat every row as data
    else:
        body = rows[1:]
    return run_batch([r[col] for r in body if len(r) > col], "csv", save=True)


@app.get("/history")
def get_history(limit: int = 50):
    """Most recent analyses, newest first."""
    return db.history(min(limit, 500))


@app.get("/stats")
def get_stats():
    """Aggregates for the dashboard: totals, aspect averages, daily trend."""
    return db.stats()


@app.get("/export.csv")
def export_csv():
    """Download the full analysis history as a CSV file."""
    aspect_names = list(ASPECTS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "source", "text", "sentiment", "confidence",
                "score", "needs_review", *aspect_names])
    for r in db.all_rows():
        w.writerow([r["id"], r["created_at"], r["source"], r["text"], r["sentiment"],
                    r["confidence"], r["score"], r["needs_review"],
                    *[r["aspects"].get(a, "") for a in aspect_names]])
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cinefeels_history.csv"},
    )


@app.delete("/history")
def clear_history():
    """Delete all stored analyses."""
    db.clear()
    return {"status": "cleared"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the frontend HTML page."""
    with open("index.html", "r") as f:
        return f.read()
