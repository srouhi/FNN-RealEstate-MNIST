import numpy as np
from model import (
    model_compile, forward_pass, model_update, model_gradients,
    mse_loss, ReLU, ReLU_derivative
)

data = np.loadtxt("LincolnHomeSales.csv", delimiter=",", skiprows=1)
X_train = data[:, :-1]
Y_train = data[:, -1]

# Normalize
for i in range(X_train.shape[1]):
    max_val = np.max(X_train[:, i])
    if max_val != 0:
        X_train[:, i] /= max_val

Y_train /= np.max(Y_train)

#params
num_input = X_train.shape[1]
num_hidden = 8
num_output = 1
activation = ReLU
activation_derivative = ReLU_derivative
learning_rate = 0.01
epochs = 100
batch_size = 32


W2, b2, W3, b3 = model_compile(num_input, num_hidden, num_output)

# training
for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    # Mini-batch gradient descent
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_train[start:end]
        Y_batch = Y_train[start:end]
        dW2, db2, dW3, db3 = model_gradients(
            X_batch, Y_batch, W2, b2, W3, b3, activation, activation_derivative
        )
        W2, b2, W3, b3 = model_update(W2, b2, W3, b3, dW2, db2, dW3, db3, learning_rate)

    # Evaluate performance 
    if (epoch + 1) % 10 == 0:
        total_loss = 0
        correct = 0
        for x, y in zip(X_train, Y_train):
            _, _, _, pred = forward_pass(x, W2, b2, W3, b3, activation)
            loss = mse_loss(y, pred)
            total_loss += loss
            if abs(pred - y) < 0.05:  
                correct += 1
        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
