#loss func
import numpy as np

def mse_loss(y_true, y_pred):
    """Mean Squared Error Loss"""
    return 0.5 * np.linalg.norm(y_true - y_pred) ** 2

def categorical_crossentropy(y_true, y_pred):
    """Categorical Crossentropy Loss for classification tasks"""
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# Activation and derivatives
def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # stability fix
    return e_x / np.sum(e_x)

def softmax_derivative(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

#Data normalization
def normalize_features(X):
    """Normalize each column of X to [0, 1] range"""
    X_norm = np.copy(X)
    for i in range(X.shape[1]):
        max_val = np.max(X[:, i])
        if max_val != 0:
            X_norm[:, i] /= max_val
    return X_norm

def normalize_targets(Y):
    """Normalize target values to [0, 1] range"""
    max_val = np.max(Y)
    return Y / max_val if max_val != 0 else Y

#calculate accuracy
def regression_accuracy(y_true, y_pred, tolerance=0.05):
    """Return % of predictions within a tolerance range of the true values"""
    correct = np.sum(np.abs(y_pred - y_true) < tolerance)
    return correct / len(y_true)


