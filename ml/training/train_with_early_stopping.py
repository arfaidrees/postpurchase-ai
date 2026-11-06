
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "customer_requests.csv"

torch.manual_seed(42)
np.random.seed(42)

# 1. Load data
df = pd.read_csv(DATA_PATH)
texts = df["message"].astype(str).str.lower().str.strip()
labels = df["category"].astype(str).str.strip()

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# 2. Preserve the original 80/20 train/test split
train_text, test_text, train_y, test_y = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Split training portion into train and validation
train_text, val_text, train_y, val_y = train_test_split(
    train_text,
    train_y,
    test_size=0.25,
    random_state=42,
    stratify=train_y
)

print("Training samples:", len(train_text))
print("Validation samples:", len(val_text))
print("Held-out test samples:", len(test_text))

# 4. Fit TF-IDF on training subset only
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(train_text).toarray()
X_val = vectorizer.transform(val_text).toarray()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(train_y, dtype=torch.long)
y_val = torch.tensor(val_y, dtype=torch.long)


class IntentNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = IntentNetwork(
    input_size=X_train.shape[1],
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0.001
)

best_val_loss = float("inf")
best_model_state = None
best_epoch = 0

patience = 15
min_delta = 0.0001
epochs_without_improvement = 0

training_history = []
validation_history = []

# 5. Training loop
for epoch in range(1, 201):

    model.train()

    optimizer.zero_grad()

    train_logits = model(X_train)
    train_loss = criterion(train_logits, y_train)

    train_loss.backward()
    optimizer.step()

    # Evaluate on validation data
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)

    train_loss_value = train_loss.item()
    val_loss_value = val_loss.item()

    training_history.append(train_loss_value)
    validation_history.append(val_loss_value)

    if val_loss_value < best_val_loss - min_delta:
        best_val_loss = val_loss_value
        best_epoch = epoch
        best_model_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss_value:.4f} | "
            f"Val Loss: {val_loss_value:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

# 6. Restore best model weights
model.load_state_dict(best_model_state)

print("\nBest epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))

# 7. Evaluate on validation data
model.eval()

with torch.no_grad():
    predictions = model(X_val).argmax(dim=1)
    accuracy = (predictions == y_val).float().mean().item()

print("Validation accuracy:", round(accuracy, 4))

# 8. Save loss history for visualization
history = pd.DataFrame({
    "epoch": range(1, len(training_history) + 1),
    "train_loss": training_history,
    "val_loss": validation_history
})

output_path = ROOT / "ml" / "evaluation" / "training_history.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
history.to_csv(output_path, index=False)

print("Training history saved to:", output_path)
