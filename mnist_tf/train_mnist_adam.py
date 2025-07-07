import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#trying with Adam optimizer
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = tf.reshape(train_images, [60000, 784])
test_images = tf.reshape(test_images, [10000, 784])

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_function = tf.losses.CategoricalCrossentropy
optimizer = tf.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

model.fit(train_images,
          train_labels, 
          epochs=10, 
          batch_size=32)