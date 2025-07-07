import numpy as np

# Loss Func
def mse_loss(y_true, y_pred):
    return 0.5 * np.linalg.norm(y_true - y_pred) ** 2

def mse_loss_grad(y_true, y_pred):
    return y_pred - y_true

# Activation Func
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return (x > 0).astype(float)

# Model Initialization 
def model_compile(num_input, num_hidden, num_output):
    W2 = np.random.rand(num_hidden, num_input)
    b2 = np.random.rand(num_hidden)
    W3 = np.random.rand(num_output, num_hidden)
    b3 = np.random.rand(num_output)
    return W2, b2, W3, b3

# Forward Pass
def forward_pass(X, W2, b2, W3, b3, activation):
    Z2 = np.dot(W2, X) + b2
    A2 = activation(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = activation(Z3)
    return Z2, A2, Z3, A3

# Backprop
def backpropagation(X, Y, W2, b2, W3, b3, activation, activation_derivative):
    Z2, A2, Z3, A3 = forward_pass(X, W2, b2, W3, b3, activation)
    d3 = mse_loss_grad(Y, A3) * activation_derivative(Z3)
    dW3 = np.outer(d3, A2)
    db3 = d3
    d2 = np.dot(W3.T, d3) * activation_derivative(Z2)
    dW2 = np.outer(d2, X)
    db2 = d2
    return dW2, db2, dW3, db3

#Gradients over Dataset 
def model_gradients(X_train, Y_train, W2, b2, W3, b3, activation, activation_derivative):
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dW3 = np.zeros_like(W3)
    db3 = np.zeros_like(b3)

    for x, y in zip(X_train, Y_train):
        temp_dW2, temp_db2, temp_dW3, temp_db3 = backpropagation(x, y, W2, b2, W3, b3, activation, activation_derivative)
        dW2 += temp_dW2
        db2 += temp_db2
        dW3 += temp_dW3
        db3 += temp_db3

    num = len(X_train)
    return dW2 / num, db2 / num, dW3 / num, db3 / num

# Update Weights
def model_update(W2, b2, W3, b3, dW2, db2, dW3, db3, learning_rate):
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W2, b2, W3, b3
