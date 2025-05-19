import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Paths
MFCC_DIR = "Data/mfcc"
metadata_path = os.path.join(MFCC_DIR, "metadata.csv")

# Load metadata
df = pd.read_csv(metadata_path)
label_names = sorted(df["label"].unique())
label_to_idx = {name: i for i, name in enumerate(label_names)}

# Load MFCC features
X = []
y = []

for _, row in df.iterrows():
    mfcc = np.load(row["file"])
    X.append(mfcc.T)  # shape (time, features)
    y.append(label_to_idx[row["label"]])

X = np.array(X)  # shape (samples, time, features)
y = to_categorical(np.array(y), num_classes=len(label_names))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)

# Build model: Conv1D → LSTM → Dense
model = Sequential([
    Conv1D(64, kernel_size=5, activation="relu", input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(64, return_sequences=False),
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

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.2%}")

# Plot learning curves
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend(); plt.title("Accuracy")

plt.tight_layout()
plt.show()

# Save model
os.makedirs("Models", exist_ok=True)
model.save("Models/audio_emotion_model.h5")
print("✅ Model saved to Models/audio_emotion_model.h5")
