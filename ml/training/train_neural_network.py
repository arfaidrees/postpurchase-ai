
import torch
from torch import nn

torch.manual_seed(42)

# Four features and four intent classes
X = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # tracking
    [0.0, 1.0, 0.0, 0.0],  # return
    [0.0, 0.0, 1.0, 0.0],  # refund
    [0.0, 0.0, 0.0, 1.0],  # cancellation
], dtype=torch.float32)

y = torch.tensor([0, 1, 2, 3], dtype=torch.long)


class IntentNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.network(x)


model = IntentNeuralNetwork()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.05
)

# Training
model.train()

for epoch in range(1, 201):

    # Step 1: Clear old gradients
    optimizer.zero_grad()

    # Step 2: Forward propagation
    logits = model(X)

    # Step 3: Calculate loss
    loss = criterion(logits, y)

    # Step 4: Backpropagation
    loss.backward()

    # Step 5: Update parameters
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f}"
        )

# Evaluate on the same examples
model.eval()

with torch.no_grad():
    logits = model(X)
    predictions = torch.argmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)

print("\nActual labels:", y.tolist())
print("Predicted labels:", predictions.tolist())
print("\nProbabilities:")
print(probabilities)
