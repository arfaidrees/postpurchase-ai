
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "store_policies.csv"

# Load knowledge base
df = pd.read_csv(DATA_PATH)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine titles and policy content
documents = (
    df["title"].astype(str)
    + ". "
    + df["content"].astype(str)
).tolist()

# Create embeddings for all policies
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True,
    normalize_embeddings=True
)


def search_policies(query: str, top_k: int = 3):
    # Convert customer question into an embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Find the most similar policy documents
    hits = util.semantic_search(
        query_embedding,
        document_embeddings,
        top_k=min(top_k, len(documents))
    )[0]

    results = []

    for hit in hits:
        index = hit["corpus_id"]

        results.append({
            "title": df.iloc[index]["title"],
            "content": df.iloc[index]["content"],
            "similarity": round(float(hit["score"]), 4)
        })

    return results


if __name__ == "__main__":
    questions = [
        "How many days do I have to send something back?",
        "When will I get my money back?",
        "Where can I check my package location?",
        "My product arrived broken. What should I do?"
    ]

    for question in questions:
        print(f"\nCustomer: {question}")

        results = search_policies(question, top_k=2)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['title']} "
                f"(similarity={result['similarity']})"
            )
            print(f"   {result['content']}")
