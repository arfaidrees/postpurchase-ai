
class CustomerRequest:
    def __init__(
        self,
        customer_id: int,
        order_id: str,
        message: str
    ):
        self.customer_id = customer_id
        self.order_id = order_id
        self.message = message
        self.status = "pending"

    def display(self):
        print(f"Customer: {self.customer_id}")
        print(f"Order: {self.order_id}")
        print(f"Message: {self.message}")
        print(f"Status: {self.status}")


request1 = CustomerRequest(
    101,
    "ORD-1001",
    "Where is my order?"
)

request2 = CustomerRequest(
    102,
    "ORD-1002",
    "I want a refund"
)

request1.display()
print("----------------")
request2.display()
