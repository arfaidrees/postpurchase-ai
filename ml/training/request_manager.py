
from dataclasses import dataclass, field


@dataclass
class CustomerRequest:
    customer_id: int
    order_id: str
    message: str
    status: str = "pending"


@dataclass
class RequestManager:
    requests: list[CustomerRequest] = field(
        default_factory=list
    )

    def add_request(self, request: CustomerRequest):
        self.requests.append(request)

    def get_pending_requests(self):
        return [
            request
            for request in self.requests
            if request.status == "pending"
        ]

    def get_requests_by_customer(self, customer_id: int):
        return [
            request
            for request in self.requests
            if request.customer_id == customer_id
        ]

    def resolve_request(self, order_id: str):
        for request in self.requests:
            if request.order_id == order_id:
                request.status = "resolved"
                return True

        return False

    def count_requests_by_status(self):
        counts = {}

        for request in self.requests:
            status = request.status

            if status in counts:
                counts[status] += 1
            else:
                counts[status] = 1

        return counts


# Create the request manager
manager = RequestManager()

# Add three customer requests
manager.add_request(
    CustomerRequest(
        101, "ORD-1001", "Where is my order?"
    )
)

manager.add_request(
    CustomerRequest(
        102, "ORD-1002", "I need a refund"
    )
)

manager.add_request(
    CustomerRequest(
        101, "ORD-1003", "Cancel my order",
        status="resolved"
    )
)

# Count requests before resolving
print("Before:", manager.count_requests_by_status())

# Resolve one request
result = manager.resolve_request("ORD-1001")

print("Request resolved:", result)

# Count requests after resolving
print("After:", manager.count_requests_by_status())

# Try an order that doesn't exist
result = manager.resolve_request("ORD-9999")

print("Unknown order resolved:", result)
