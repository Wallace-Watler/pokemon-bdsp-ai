"""Train a neural network to classify battle UI images."""

import sys

if not len(sys.argv) == 5:
    print("Usage: python train_ui.py <ui_folder> <scale> <scene> <epochs>")
    exit(0)

import tensorflow as tf
import numpy as np
import cv2
import csv
import json
import ism
import finite_state_machine as fsm
import os
import random

SCALE = int(sys.argv[2])
UI_FOLDER = sys.argv[1]
SCALE_FOLDER = f"{UI_FOLDER}/training_data/scale{SCALE}"
SCENE_FOLDER = f"{SCALE_FOLDER}/{sys.argv[3]}"
MODEL_FOLDER = f"{SCALE_FOLDER}/trained_model"
EPOCHS = int(sys.argv[4])

# Load data
states = fsm.load_states(UI_FOLDER + "/states.csv")
states = [state.name for state in states]

print("Loading images...")
labels = ism.load_labels(SCENE_FOLDER)
images = [np.array(cv2.imread(f"{SCENE_FOLDER}/{label[0]}.png")) for label in labels]
labels = [states.index(label[1]) for label in labels]

# Randomly split into training and test sets
indices = [*range(0, len(images))]
random.shuffle(indices)
split_at = 9 * len(images) // 10
train_images = np.asarray([images[i] for i in indices[:split_at]])
test_images = np.asarray([images[i] for i in indices[split_at:]])
train_labels = np.asarray([labels[i] for i in indices[:split_at]])
test_labels = np.asarray([labels[i] for i in indices[split_at:]])

# Normalize image data to [0.0, 1.0]
train_images = train_images / 255.0
test_images = test_images / 255.0
print("Preprocessing complete.")

# Load and train model
model = None
if os.path.exists(MODEL_FOLDER):
    model = tf.keras.models.load_model(MODEL_FOLDER)
else:
    size = ism.load_size(UI_FOLDER)
    width = size["width"] // SCALE
    height = size["height"] // SCALE

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(height, width, 3)),
        tf.keras.layers.Dense(len(states))
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

model.fit(train_images, train_labels, epochs = EPOCHS)

# Evaluate and save model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)
model.save(MODEL_FOLDER)
print("Model saved.")
