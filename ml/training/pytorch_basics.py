
import torch

# A scalar: one number
scalar = torch.tensor(5.0)

# A vector: one-dimensional tensor
vector = torch.tensor([2.0, 4.0, 6.0])

# A matrix: two-dimensional tensor
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("Scalar shape:", scalar.shape)
print("Vector shape:", vector.shape)
print("Matrix shape:", matrix.shape)

# Tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("Addition:", a + b)
print("Element-wise multiplication:", a * b)
print("Dot product:", torch.dot(a, b))

# Three customer messages, four illustrative features each
features = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

print("Features shape:", features.shape)
print("First message:", features[0])
print("Feature totals:", features.sum(dim=0))
