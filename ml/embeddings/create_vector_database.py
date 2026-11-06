
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "store_policies.csv"
DB_PATH = ROOT / "ml" / "vector_db"

# 1. Load policy data
df = pd.read_csv(DATA_PATH)

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Create a persistent ChromaDB client
client = chromadb.PersistentClient(path=str(DB_PATH))

# 4. Create or open a collection
collection = client.get_or_create_collection(
    name="store_policies",
    metadata={"hnsw:space": "cosine"}
)

# 5. Prepare documents and metadata
documents = (
    df["title"].astype(str)
    + ". "
    + df["content"].astype(str)
).tolist()

ids = [str(i) for i in df["id"]]

metadata = [
    {"title": str(title)}
    for title in df["title"]
]

# 6. Generate embeddings
embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()

# 7. Insert or update records
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata
)

print("Vector database created successfully.")
print("Stored documents:", collection.count())
print("Database path:", DB_PATH)
