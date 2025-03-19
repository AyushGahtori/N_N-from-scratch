from layer import Layer
import numpy as np

class Dense(Layer):
    # Constructor for the Dense layer
    # input_size: number of input neurons
    # output_size: number of output neurons
    def __init__(self, input_size, output_size):
        # Initialize weights and biases with random values
        self.weights = np.random.randn(output_size, input_size) # weights matrix = j x i
        self.biases = np.random.randn(output_size, 1) # biases matrix = j x 1
    def forward(self, input):
        # Compute output = weights*input + biases
        # Store input for backward pass
        # Y = W.X + b (dot product)
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases 

    def backward(self, output_gradient, learning_rate):
        # Compute dE/dW, dE/dX, dE/db
        # dE/dW = dE/dY * dY/dW = dE/dY * X
        # dE/dX = dE/dY * dY/dX = dE/dY * W
        # dE/db = dE/dY * dY/db = dE/dY
        # Update weights and biases
        # dE/dY = output_gradient
        # dE/dW = dE/dY * X
        # dE/db = dE/dY
        # dE/dX = W^T * dE/dY
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        return np.dot(self.weights.T, output_gradient)       