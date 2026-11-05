# PostPurchase AI - Day 1
# Lesson 1: Python Fundamentals

customer_request = {
    "customer_id": 101,
    "message": "Where is my order?",
    "order_id": "ORD-1001",
    "status": "pending"
}

print(customer_request["message"])


def detect_keyword(message: str) -> str:
    message = message.lower()

    if "refund" in message:
        return "refund"

    if "return" in message:
        return "return"

    if "cancel" in message:
        return "cancel"

    if "order" in message:
        return "tracking"

    return "unknown"


result = detect_keyword(customer_request["message"])

print("Detected intent:", result)
