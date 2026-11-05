
from pathlib import Path
import re

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# 1. Define project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "ml" / "data" / "customer_requests.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Text preprocessing
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# 3. Load and validate dataset
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["message", "category"])
df["message"] = df["message"].astype(str).apply(clean_text)
df["category"] = df["category"].astype(str).str.strip()

df = df.drop_duplicates(subset=["message"])

print("Total samples:", len(df))
print("\nClass distribution:")
print(df["category"].value_counts())


# 4. Separate features and labels
X = df["message"]
y = df["category"]


# 5. Split before learning vocabulary
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 6. Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])


# 7. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

print("Training complete!")


# 8. Evaluate on unseen test examples
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_,
    )
)

print("\nClass order:")
print(model.classes_)


# 9. Save trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")


# 10. Test new customer messages
sample_messages = [
    "Where is my delivery?",
    "Please give me my money back",
    "I need to send these shoes back",
    "Stop my order before it ships",
]

print("\nSample predictions:")

for message in sample_messages:
    cleaned = clean_text(message)

    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]

    confidence = max(probabilities)

    print(
        f"Message: {message}\n"
        f"Prediction: {prediction}\n"
        f"Model probability: {confidence:.2%}\n"
    )
