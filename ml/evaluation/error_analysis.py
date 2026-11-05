
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "ml/data/customer_requests.csv"
MODEL_PATH = ROOT / "ml/models/intent_classifier.joblib"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["message", "category"])
df["message"] = (
    df["message"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
df["category"] = df["category"].astype(str).str.strip()
df = df.drop_duplicates(subset=["message"])

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["category"],
    test_size=0.20,
    random_state=42,
    stratify=df["category"],
)

model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "message": X_test.to_numpy(),
    "actual": y_test.to_numpy(),
    "predicted": predictions,
})

results["correct"] = (
    results["actual"] == results["predicted"]
)

print("\nAll test predictions:")
print(results.to_string(index=False))

print("\nIncorrect predictions:")
print(
    results[~results["correct"]].to_string(index=False)
)

print(
    "\nAccuracy:",
    accuracy_score(y_test, predictions)
)
