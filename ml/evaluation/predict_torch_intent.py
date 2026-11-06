
from pathlib import Path

import joblib
import torch

from ml.models.torch_intent_model import TextIntentNetwork

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"

# Load model checkpoint
checkpoint = torch.load(
    MODEL_DIR / "torch_intent_classifier.pt",
    map_location="cpu",
    weights_only=True
)

model = TextIntentNetwork(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load fitted vectorizer
vectorizer = joblib.load(
    MODEL_DIR / "torch_tfidf_vectorizer.joblib"
)

classes = checkpoint["classes"]


def predict_intent(message: str):
    cleaned = " ".join(message.lower().split())

    features = vectorizer.transform([cleaned]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "message": message,
        "intent": classes[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()), 4
        ),
        "probabilities": {
            label: round(float(prob.item()), 4)
            for label, prob in zip(classes, probabilities)
        }
    }


if __name__ == "__main__":
    messages = [
        "Where is my order?",
        "I want to return these shoes",
        "Please refund my money",
        "Cancel my purchase"
    ]

    for message in messages:
        print(predict_intent(message))
