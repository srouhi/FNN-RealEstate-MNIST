import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = tf.reshape(train_images, [60000, 784])
test_images = tf.reshape(test_images, [10000, 784])

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with SGD optimizer
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train model
model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels)
)
