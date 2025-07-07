
# 🧠 Feedforward Neural Network Experiments

This repository contains implementations and training experiments for feedforward neural networks (FNNs) using both **custom NumPy-based models** and **TensorFlow Keras** models.

---

## 📁 Project Structure


```text
.
├── model.py                 # Custom FNN logic (forward pass, backprop, training loop)
├── train.py                 # Trains the custom NumPy model on Lincoln Home Sales data
├── train_mnist_sgd.py       # Trains a Keras model on MNIST (SGD optimizer)
├── train_mnist_adam.py      # Trains a Keras model on MNIST (Adam optimizer)
├── train_fashion_mnist.py   # Trains a Keras model on Fashion-MNIST (Adam optimizer)
├── utils.py                 # Utility functions (activation functions, loss, data prep)
├── LincolnHomeSales.csv     # Dataset of single-family home sales in Lincoln, NE
└── README.md                # This file
```
---

## 📊 Datasets Used

### 🏠 Lincoln Home Sales
- Real estate dataset of homes sold in Lincoln, NE (2016–2022)
- Used with a NumPy-built FNN to predict home prices

### ✍️ MNIST
- Classic handwritten digits dataset (0–9)
- Trained using both SGD and Adam optimizers in TensorFlow

### 👗 Fashion-MNIST
- Drop-in replacement for MNIST with clothing items (t-shirts, shoes, bags, etc.)
- Used to explore overfitting and training behavior

---

## ⚙️ Features

### ✅ Custom NumPy-Based Neural Network
- Manual implementation of:
  - Forward pass
  - Backpropagation
  - ReLU + Sigmoid activations
  - Loss and gradients
- Batch training with configurable learning rate and metrics

### ✅ TensorFlow Experiments
- Keras models for fast experimentation
- Optimizer comparison: SGD vs. Adam
- Loss monitoring and validation
- Fashion-MNIST overfitting investigation

---

## 🧪 How to Run

### Custom FNN:
```bash
python train.py
````

### Keras-based Models:

```bash
python train_mnist_sgd.py
python train_mnist_adam.py
python train_fashion_mnist.py
```

---

## 📈 Results & Insights

* **Adam** optimizer often converges faster than **SGD**, especially in early epochs
* **SGD** sometimes overtakes Adam in later training stages (depending on the data)
* **Fashion-MNIST** is prone to overfitting beyond 10–15 epochs
* For **Lincoln Home Sales**, normalization significantly improves convergence

---

## 📌 Future Improvements

* Add early stopping and validation set handling in the NumPy model
* Visualize loss and accuracy curves
* Implement support for more hidden layers
* Save model weights and predictions

---

## 🧑‍💻 Author

**Shay -Shaghayegh- Rouhi**
Data Science | AI | ML
