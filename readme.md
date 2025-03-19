
# Neural Networks from Scratch - Theory and Mathematical Foundations

## 1. Neuron Structure
A single neuron computes:
```
output = activation_function(Σ(weights * inputs) + bias)
```

## 2. Activation Functions

### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) * (1 - f(x))
```

### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - tanh^2(x)
```

## 3. Forward Propagation
For each layer:
```
Z = W * A + b
A = activation_function(Z)
```
Where:
- W: Weight matrix
- A: Input/activation matrix
- b: Bias vector
- Z: Pre-activation output

## 4. Backpropagation

### Chain Rule Application
```
dL/dW = dL/dA * dA/dZ * dZ/dW
dL/db = dL/dA * dA/dZ * dZ/db
```

### Layer-wise Calculations
For output layer:
```
dZ = A - Y (for MSE loss)
dW = (1/m) * dZ * A_prev.T
db = (1/m) * Σ(dZ)
```

For hidden layers:
```
dZ = W_next.T * dZ_next ⊙ activation_function'(Z)
dW = (1/m) * dZ * A_prev.T
db = (1/m) * Σ(dZ)
```

## 5. Loss Functions

### Mean Squared Error (MSE)
```
L = (1/m) * Σ(y_pred - y_true)²
dL/dy_pred = (2/m) * (y_pred - y_true)
```

### Binary Cross-Entropy
```
L = -(1/m) * Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
dL/dy_pred = -(y_true/y_pred) + (1-y_true)/(1-y_pred)
```

## 6. Weight Initialization

### Xavier/Glorot Initialization
```
W = np.random.randn(n_out, n_in) * sqrt(1/n_in)
```

### He Initialization
```
W = np.random.randn(n_out, n_in) * sqrt(2/n_in)
```

## 7. Optimization Algorithms

### Gradient Descent
```
W = W - learning_rate * dW
b = b - learning_rate * db
```

### Momentum
```
V_w = beta * V_w + (1-beta) * dW
V_b = beta * V_b + (1-beta) * db
W = W - learning_rate * V_w
b = b - learning_rate * V_b
```

### Adam
```
m_w = beta1 * m_w + (1-beta1) * dW
v_w = beta2 * v_w + (1-beta2) * dW²
m_w_corrected = m_w / (1-beta1^t)
v_w_corrected = v_w / (1-beta2^t)
W = W - learning_rate * m_w_corrected / (sqrt(v_w_corrected) + epsilon)
```

## 8. Regularization

### L2 Regularization
```
L = original_loss + (lambda/2m) * Σ(W²)
dW = original_dW + (lambda/m) * W
```

### Dropout
During training:
```
mask = np.random.rand(*A.shape) < keep_prob
A = A * mask
A = A / keep_prob  # Scale to maintain expected value
```

## 9. Batch Normalization
```
μ = (1/m) * Σ(x)
σ² = (1/m) * Σ((x-μ)²)
x_norm = (x-μ) / sqrt(σ² + ε)
out = γ * x_norm + β
```

## 10. Learning Rate Decay
```
learning_rate = initial_learning_rate / (1 + decay_rate * epoch_num)
```

These formulas and concepts form the mathematical foundation of neural networks and are essential for understanding their implementation from scratch.
