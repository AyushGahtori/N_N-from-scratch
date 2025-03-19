import numpy as np

def mse(y_true, y_pred):
    # Mean Squared Error
    return np.mean(np.power(y_true-y_pred, 2))
    # MSE = (1/n) * Σ(y_true - y_pred)²
def mse_prime(y_true, y_pred):
    # Derivative of Mean Squared Error
    return 2*(y_pred-y_true)/y_true.size
    # dMSE = 2/n * (y_pred - y_true)