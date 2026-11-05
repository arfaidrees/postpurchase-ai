
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "customer_requests.csv"

df = pd.read_csv(DATA_PATH)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nCategory counts:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
