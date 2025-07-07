# Feedforward Neural Network Experiments

This repository contains implementations and training experiments for feedforward neural networks (FNNs) using both **custom NumPy-based models** and **TensorFlow Keras** models.

---

## Project Structure


```text
.
├── real_estate_fnn/
|  ├── model.py                 # Custom FNN logic (forward pass, backprop, training loop)
|  ├── train.py                 # Trains the custom NumPy model on Lincoln Home Sales data
|  ├── utils.py                 # Utility functions (activation functions, loss, data prep)
|  ├── LincolnHomeSales.csv     # Dataset of single-family home sales in Lincoln, NE
├── mnist_tf/
|  ├── train_mnist_sgd.py       # Trains a Keras model on MNIST (SGD optimizer)
|  ├── train_mnist_adam.py      # Trains a Keras model on MNIST (Adam optimizer)
|  ├── train_fashion_mnist.py   # Trains a Keras model on Fashion-MNIST (Adam optimizer)
└── README.md                   # This file
```
---

## Datasets Used

### Lincoln Home Sales
- Real estate dataset of homes sold in Lincoln, NE (2016–2022)
- Used with a NumPy-built FNN to predict home prices

### MNIST
- Classic handwritten digits dataset (0–9)
- Trained using both SGD and Adam optimizers in TensorFlow

### Fashion-MNIST
- Drop-in replacement for MNIST with clothing items (t-shirts, shoes, bags, etc.)
- Used to explore overfitting and training behavior

---

## Features

### Custom NumPy-Based Neural Network
- Manual implementation of:
  - Forward pass
  - Backpropagation
  - ReLU + Sigmoid activations
  - Loss and gradients
- Batch training with configurable learning rate and metrics

### TensorFlow Experiments
- Keras models for fast experimentation
- Optimizer comparison: SGD vs. Adam
- Loss monitoring and validation
- Fashion-MNIST overfitting investigation

---

## How to Run

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

## Results & Insights

### Optimizer Behavior (MNIST)
* **Adam** optimizer converged more quickly than **SGD** in the early stages (first 5–10 epochs).
* **SGD** sometimes achieved **lower final loss** than Adam in later training stages, despite slower start.
* **Accuracy:** Adam consistently maintained **higher classification accuracy** across all epochs.

### Overfitting in Fashion-MNIST
* Clear signs of **overfitting** emerged after ~10–15 epochs.
* The **training loss continued to decrease**, but **validation loss plateaued or increased** after epoch 10.
* Best performance was typically achieved by **epoch 7–10**, making this a reasonable early stopping point.

### Lincoln Home Sales (Custom NumPy FNN)
* **Data normalization** had a significant impact on training stability and convergence.
* Model was able to **learn meaningful patterns** using basic ReLU activations and MSE loss.
* Accuracy was approximated by checking if predictions were **within a small error threshold**.
* **Batch size and learning rate tuning** affected training quality, with `batch_size=64` and `lr=0.01` showing strong performance.

---

These insights suggest:
- Adam is a better default for early training.
- Validation monitoring is essential to avoid overfitting.
- Even a basic NumPy FNN can extract signal from real-world data with proper prep and tuning.

---

## Author  
**Shay -Shaghayegh- Rouhi**
Data Science | AI | ML
