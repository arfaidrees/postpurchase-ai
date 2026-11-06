
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

messages = [
    "Where is my order?",
    "Can you track my package?",
    "I want a refund",
    "Please give my money back",
    "How do I return my shoes?",
]

# Convert messages into embeddings
embeddings = model.encode(
    messages,
    convert_to_tensor=True
)

print("Total messages:", len(messages))
print("Embedding shape:", embeddings.shape)

print("\nFirst 10 embedding values:")
print(embeddings[0][:10])

# Compare message meanings using cosine similarity
similarities = cos_sim(embeddings, embeddings)

print("\nCosine similarity matrix:")
print(similarities.round(decimals=3))

print("\nSimilarity examples:")

pairs = [
    (0, 1),  # Tracking vs tracking
    (0, 2),  # Tracking vs refund
    (2, 3),  # Refund vs refund
    (2, 4),  # Refund vs return
]

for i, j in pairs:
    score = similarities[i][j].item()

    print(f"\nMessage 1: {messages[i]}")
    print(f"Message 2: {messages[j]}")
    print(f"Similarity: {score:.4f}")
