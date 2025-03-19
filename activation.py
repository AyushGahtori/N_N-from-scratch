from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        # activation: The activation function (e.g. sigmoid, tanh, ReLU)
        # activation_prime: The derivative of the activation function
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # Forward propagation: y = f(x)
        # where f is the activation function
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Backward propagation using chain rule:
        # dE/dx = dE/dy * dy/dx
        # where dE/dy is output_gradient
        # and dy/dx is activation_prime(input) (dy/dx is the derivative of the activation function)
        return np.multiply(output_gradient, self.activation_prime(self.input))