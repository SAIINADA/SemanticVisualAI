import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("Loading face emotion dataset...")

# dataset paths
train_dir = "datasets/image_emotion/train"
test_dir = "datasets/image_emotion/test"

img_height = 48
img_width = 48
batch_size = 32

# load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

# load testing dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

print("Dataset loaded successfully")

# improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# build CNN model
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

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# create models folder if missing
os.makedirs("models", exist_ok=True)

# save model
model.save("models/face_emotion_model.h5")

print("Face emotion model saved successfully in models/")