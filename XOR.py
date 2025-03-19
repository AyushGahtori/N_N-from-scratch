import numpy as np  # Import numpy library for numerical computations

from dense import Dense  # Import Dense layer class for creating fully connected layers
from activations import Tanh  # Import Tanh activation function (f(x) = (e^x - e^-x)/(e^x + e^-x))
from losses import mse, mse_prime  # Import Mean Squared Error loss (L = 1/n * Σ(y - ŷ)²) and its derivative
from network import train  # Import training function

# Input data for XOR problem: reshape to (samples, features, 1)
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
# Target outputs for XOR: reshape to (samples, outputs, 1)
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Define neural network architecture
network = [
    Dense(2, 3),  # Input layer: 2 neurons → 3 neurons, W(2x3) and b(3x1)
    Tanh(),       # Activation: f(x) = tanh(x)
    Dense(3, 1),  # Hidden layer: 3 neurons → 1 output neuron, W(3x1) and b(1x1)
    Tanh()        # Output activation: f(x) = tanh(x)
]

epochs = 10000        # Number of training iterations
learning_rate = 0.1   # Step size for gradient descent: w = w - η∇L

# Training loop
for e in range(epochs):
    error = 0  # Initialize error for this epoch
    for x, y in zip(X, Y):
        # Forward propagation: compute a^(l) = f(W^(l)a^(l-1) + b^(l))
        output = x  # Initial input a^(0)
        for layer in network:
            output = layer.forward(output)  # Each layer computes: z = Wx + b, a = f(z)

        # Compute error using MSE: L = 1/n * Σ(y - ŷ)²
        error += mse(y, output)

        # Backward propagation (chain rule): ∂L/∂W^(l) = ∂L/∂a^(l) * ∂a^(l)/∂z^(l) * ∂z^(l)/∂W^(l)
        grad = mse_prime  # Start with loss derivative: ∂L/∂a^(L) = -2(y - ŷ)/n
        for layer in reversed(network):
            # Each layer computes gradients and updates parameters:
            # W = W - η * ∂L/∂W
            # b = b - η * ∂L/∂b
            grad = layer.backward(grad, learning_rate)

    error /= len(X)  # Average error over all samples: L = 1/n * Σ L_i
    if e % 1000 == 0:
        print(f"epoch: {e}, error: {error}")  # Print training progress every 1000 epochs