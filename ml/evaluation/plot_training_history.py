
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
history_path = ROOT / "ml" / "evaluation" / "training_history.csv"

df = pd.read_csv(history_path)

plt.figure(figsize=(9, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss"
)

best_row = df.loc[df["val_loss"].idxmin()]

plt.axvline(
    best_row["epoch"],
    linestyle="--",
    label=f"Lowest validation loss: epoch {int(best_row['epoch'])}"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("PostPurchase AI - Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
