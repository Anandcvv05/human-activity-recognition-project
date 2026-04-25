import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import load_dataset, MAX_FRAMES, IMG_SIZE

DATASET_PATH = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "action_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "class_names.json")

CLASS_NAMES = [
    "walking",
    "jogging",
    "running",
    "boxing",
    "handclapping",
    "handwaving"
]

NUM_CLASSES = len(CLASS_NAMES)

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
X, y = load_dataset(DATASET_PATH, CLASS_NAMES)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("No videos found in dataset folders.")

y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

model = Sequential([
    TimeDistributed(
        Conv2D(32, (3, 3), activation="relu"),
        input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    ),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Conv2D(64, (3, 3), activation="relu")),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Conv2D(128, (3, 3), activation="relu")),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Flatten()),

    LSTM(128, return_sequences=False),
    Dropout(0.5),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=4,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

with open(LABEL_PATH, "w") as f:
    json.dump(CLASS_NAMES, f)

print(f"Model saved to: {MODEL_PATH}")
print(f"Class names saved to: {LABEL_PATH}")