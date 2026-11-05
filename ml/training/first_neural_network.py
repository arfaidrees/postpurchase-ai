
import torch
from torch import nn

torch.manual_seed(42)

# Define our neural network
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


# Create the model
model = IntentNeuralNetwork()

print("Neural Network Architecture:")
print(model)

# Three customer messages with four example features
features = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

# Forward propagation
with torch.no_grad():
    logits = model(features)

    # Convert scores into probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Find the highest-scoring class
    predictions = torch.argmax(probabilities, dim=1)

print("\nRaw output scores:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted class indices:")
print(predictions)

print("\nProbability row sums:")
print(probabilities.sum(dim=1))
