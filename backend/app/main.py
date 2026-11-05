
from contextlib import asynccontextmanager
from pathlib import Path
import re

import joblib

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# Find the project root
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "ml" / "models" / "intent_classifier.joblib"
)


# Request schema
class IntentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


# Response schema
class IntentResponse(BaseModel):
    message: str
    intent: str
    probability: float
    probabilities: dict[str, float]


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the API starts
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Run train_classifier.py first."
        )

    app.state.model = joblib.load(MODEL_PATH)

    yield

    app.state.model = None


app = FastAPI(
    title="PostPurchase AI",
    description="AI-powered post-purchase customer support",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "PostPurchase AI",
    }


@app.post(
    "/api/v1/predict-intent",
    response_model=IntentResponse,
)
def predict_intent(
    payload: IntentRequest,
    request: Request,
):
    message = clean_text(payload.message)

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be blank.",
        )

    model = request.app.state.model

    prediction = model.predict([message])[0]

    scores = model.predict_proba([message])[0]

    probabilities = {
        str(label): round(float(score), 4)
        for label, score in zip(model.classes_, scores)
    }

    return IntentResponse(
        message=payload.message,
        intent=str(prediction),
        probability=round(
            max(probabilities.values()), 4
        ),
        probabilities=probabilities,
    )
