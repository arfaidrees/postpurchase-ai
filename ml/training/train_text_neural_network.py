
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "customer_requests.csv"

torch.manual_seed(42)
np.random.seed(42)

# Load the existing customer support dataset
df = pd.read_csv(DATA_PATH)

print("Dataset columns:", df.columns.tolist())
print("Total samples:", len(df))

# Match the columns used in your Day 1 dataset
texts = df["message"].astype(str).str.lower().str.strip()
labels = df["category"].astype(str).str.strip()
# Encode text labels as numbers
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Keep the same held-out test split as Day 1
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit TF-IDF on training data only
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class TextIntentNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = TextIntentNetwork(
    input_size=X_train.shape[1],
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train using only the training set
model.train()

for epoch in range(1, 101):
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Evaluate on unseen test data
model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    predictions = torch.argmax(test_logits, dim=1).numpy()

print("\nTest accuracy:",
      round(accuracy_score(y_test.numpy(), predictions), 4))

print("\nClassification report:")
print(classification_report(
    y_test.numpy(),
    predictions,
    target_names=encoder.classes_,
    zero_division=0
))

print("\nClass mapping:")
for index, label in enumerate(encoder.classes_):
    print(index, "->", label)
