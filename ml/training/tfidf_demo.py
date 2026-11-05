
from sklearn.feature_extraction.text import TfidfVectorizer

messages = [
    "Where is my package",
    "Track my package",
    "I want a refund",
    "Please refund my payment"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(messages)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF matrix shape:")
print(X.shape)

print("\nTF-IDF matrix:")
print(X.toarray())
