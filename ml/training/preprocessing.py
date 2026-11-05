
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


examples = [
    "  WHERE is my PACKAGE?  ",
    "I   NEED   A   REFUND",
    "Please CANCEL my order"
]

for message in examples:
    print("Original:", repr(message))
    print("Cleaned: ", repr(clean_text(message)))
    print()
