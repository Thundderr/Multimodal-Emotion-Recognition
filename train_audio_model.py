import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
MFCC_DIR = "data/mfcc"
metadata_path = os.path.join(MFCC_DIR, "metadata.csv")

# Load metadata
df = pd.read_csv(metadata_path)
label_names = sorted(df["label"].unique())
label_to_idx = {name: i for i, name in enumerate(label_names)}

# Load MFCC features
X, y = [], []
for _, row in df.iterrows():
    mfcc = np.load(row["file"])
    X.append(mfcc.T)  # shape: (time, features)
    y.append(label_to_idx[row["label"]])

X = np.array(X)
y_raw = np.array(y)
y = to_categorical(y_raw, num_classes=len(label_names))

# Split data
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_raw_train), y=y_raw_train)
class_weight_dict = dict(enumerate(class_weights))

# Build model
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Conv1D(64, kernel_size=5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(64),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(label_names), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.2%}")

# sanity check for data
sample = random.choice(df["file"].tolist())
mfcc = np.load(sample)
plt.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
plt.title(sample)
plt.colorbar()
plt.show()

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()

plt.tight_layout()
plt.show()

# Save
os.makedirs("Models", exist_ok=True)
model.save("Models/audio_emotion_model.h5")
print("✅ Model saved to Models/audio_emotion_model.h5")
