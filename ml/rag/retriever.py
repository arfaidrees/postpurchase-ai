
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "ml" / "vector_db"


class PolicyRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name="store_policies"
        )

    def retrieve(self, question: str, top_k: int = 3):
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []

        for index, document in enumerate(results["documents"][0]):
            distance = float(results["distances"][0][index])

            retrieved.append({
                "id": results["ids"][0][index],
                "title": results["metadatas"][0][index]["title"],
                "content": document,
                "distance": round(distance, 4)
            })

        return retrieved
