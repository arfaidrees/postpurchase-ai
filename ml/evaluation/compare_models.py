
from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml/data/customer_requests.csv"

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

algorithms = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
        random_state=42,
    ),
}

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)

print("Cross-validation on training data:\n")

for name, classifier in algorithms.items():
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2)
            ),
        ),
        ("classifier", classifier),
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
    )

    print(name)
    print("Fold scores:", scores.round(3))
    print("Mean F1:", round(scores.mean(), 3))
    print()
