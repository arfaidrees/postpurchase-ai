
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "ml" / "vector_db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Open existing database
client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_collection(name="store_policies")


def search_policies(query: str, top_k: int = 2):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


if __name__ == "__main__":
    questions = [
        "How long do I have to return an item?",
        "When will I receive my refund?",
        "Can I cancel an order that has shipped?"
    ]

    for question in questions:
        print(f"\nCustomer: {question}")

        results = search_policies(question)

        for index, document in enumerate(results["documents"][0]):
            title = results["metadatas"][0][index]["title"]
            distance = results["distances"][0][index]

            print(f"{index + 1}. {title}")
            print(f"   Cosine distance: {distance:.4f}")
            print(f"   {document}")
