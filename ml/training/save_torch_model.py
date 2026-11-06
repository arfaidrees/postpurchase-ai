
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.models.torch_intent_model import TextIntentNetwork

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"
DATA_PATH = ROOT / "ml" / "data" / "customer_requests.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

df = pd.read_csv(DATA_PATH)

texts = df["message"].astype(str).str.lower().str.strip()
labels = df["category"].astype(str).str.strip()

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train_text).toarray()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

model = TextIntentNetwork(
    input_size=X_train.shape[1],
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in range(1, 101):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Save weights and metadata
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
        "num_classes": len(encoder.classes_),
        "classes": encoder.classes_.tolist()
    },
    MODEL_DIR / "torch_intent_classifier.pt"
)

# Save fitted TF-IDF vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "torch_tfidf_vectorizer.joblib"
)

print("\nPyTorch model and vectorizer saved successfully.")
