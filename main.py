from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import os

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Predict whether a movie review is positive or negative.",
    version="1.0.0"
)

# Allow requests from the frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ───────────────────────────────────────────────────────────────

# Load the trained pipeline (TF-IDF + Logistic Regression) on startup
MODEL_PATH = "./notebooks/sentiment_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Please run the Jupyter notebook first to generate the model."
    )

model = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# ── Request / Response Schemas ────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    """The request body for the /predict endpoint."""
    text: str  # The movie review text to analyze

class PredictionResponse(BaseModel):
    """The response returned by the /predict endpoint."""
    sentiment: str       # "positive" or "negative"
    confidence: float    # 0.0 to 1.0 — how sure the model is
    label: str           # Human-friendly label with emoji

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check — returns OK if the server is running."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """
    Accepts a movie review text and returns the predicted sentiment.

    Example request body:
        {"text": "This movie was absolutely amazing!"}

    Example response:
        {"sentiment": "positive", "confidence": 0.97, "label": "😊 Positive"}
    """
    text = request.text.strip()

    if not text:
        return PredictionResponse(
            sentiment="neutral",
            confidence=0.0,
            label="⚠️ Please enter some text"
        )

    # The pipeline handles both TF-IDF transformation and prediction
    prediction = model.predict([text])[0]            # "positive" or "negative"
    probabilities = model.predict_proba([text])[0]    # [prob_neg, prob_pos]

    # Get confidence for the predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    # Build a friendly label
    if prediction == "positive":
        label = f"😊 Positive"
    else:
        label = f"😞 Negative"

    return PredictionResponse(
        sentiment=prediction,
        confidence=round(confidence, 4),
        label=label
    )


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the frontend HTML page."""
    with open("index.html", "r") as f:
        return f.read()
