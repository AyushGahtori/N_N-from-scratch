class Layer:
    def __init__(self):
        # Initialize input attribute to store the input data for this layer
        self.input = None
        # Initialize output attribute to store the computed output of this layer
        self.output = None

    def forward_propagation(self, input):
        # Method to compute the forward pass of the layer
        # This is a base class method that will be overridden by specific layer types
        # input: the input data to be processed
        pass

    def backward_propagation(self, output_gradient, learning_rate):
        # Method to compute the backward pass of the layer
        # This is a base class method that will be overridden by specific layer types
        # output_gradient: the gradient to propagate backward
        # learning_rate: the rate at which the layer's parameters should be updated
        pass