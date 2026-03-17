from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# PATHS â€“ adjust if needed
# -------------------------------------------------------------------
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = Path(r"C:\Users\galag\GitHub\galago_audio_project\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 40  # you can lower/raise later

# -------------------------------------------------------------------
# 1. Load datasets from folder structure
# -------------------------------------------------------------------
print("Loading datasets from:", MELS_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",   # PNGs saved via matplotlib -> RGB
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# -------------------------------------------------------------------
# 2. Performance tweaks
# -------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------------------------------------------
# 3. Define a small CNN model
# -------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),      # not super meaningful for spectrograms, but harmless
        layers.RandomRotation(0.05),
    ],
    name="data_augmentation",
)

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)

x = layers.Rescaling(1.0 / 255)(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="galago_call_cnn")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 4. Callbacks: early stopping + best model checkpoint
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / "galago_cnn_best.keras"

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
    ),
]

# -------------------------------------------------------------------
# 5. Train
# -------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# -------------------------------------------------------------------
# 6. Evaluate and save final model
# -------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.3f}")

final_model_path = OUT_DIR / "galago_cnn_final.keras"
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# -------------------------------------------------------------------
# 7. Plot training curves (optional but nice)
# -------------------------------------------------------------------
history_dict = history.history
epochs = range(1, len(history_dict["loss"]) + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict["loss"], "b-", label="Training loss")
plt.plot(epochs, history_dict["val_loss"], "r-", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict["accuracy"], "b-", label="Training acc")
plt.plot(epochs, history_dict["val_accuracy"], "r-", label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.close()
print("Saved training curves to:", OUT_DIR / "training_curves.png")

