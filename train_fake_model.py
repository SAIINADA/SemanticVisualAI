import os

# ---- FORCE CPU (disable GPU completely) ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading face emotion dataset...")

# Dataset paths
train_dir = "datasets/image_emotion/train"
test_dir = "datasets/image_emotion/test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True
)

# Load validation dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True
)

print("Dataset loaded successfully")

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Building CNN model...")

model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(48,48,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training face emotion model...")

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/face_emotion_model.h5")

print("Face emotion model saved successfully in models/")