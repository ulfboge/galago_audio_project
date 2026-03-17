from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# PATHS â€" adjust if needed
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "models" / "top7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 50  # can lower/raise later

# Only these 6 species (must match folder names exactly)
SELECTED_CLASSES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Paragalago_zanzibaricus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]


print("Selected classes:", SELECTED_CLASSES)

# -------------------------------------------------------------------
# 1. Load datasets from folder structure (filtered)
# -------------------------------------------------------------------
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
    class_names=SELECTED_CLASSES,  # <- only these classes
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
    class_names=SELECTED_CLASSES,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes actually used:", class_names)

# -------------------------------------------------------------------
# 2. Performance tweaks
# -------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------------------------------------------
# 3. Compute class weights to handle imbalance
# -------------------------------------------------------------------
def compute_class_weights(dataset, num_classes):
    counts = Counter()
    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1

    total = sum(counts.values())
    class_weights = {}
    for cls in range(num_classes):
        n = counts.get(cls, 1)
        class_weights[cls] = total / (num_classes * n)
    return class_weights, counts

class_weights, counts = compute_class_weights(train_ds, num_classes)
print("Train counts per class index:", counts)
print("Class weights:", class_weights)

# -------------------------------------------------------------------
# 4. Define CNN model (no image-level augmentation)
# -------------------------------------------------------------------
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="galago_call_cnn_top7")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 5. Callbacks
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / "galago_cnn_top7_best.keras"

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
    ),
]

# -------------------------------------------------------------------
# 6. Train
# -------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# -------------------------------------------------------------------
# 7. Evaluate and save
# -------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy (top7): {val_acc:.3f}")

final_model_path = OUT_DIR / "galago_cnn_top7_final.keras"
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# -------------------------------------------------------------------
# 8. Plot training curves
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
plt.savefig(OUT_DIR / "training_curves_top7.png", dpi=150)
plt.close()
print("Saved training curves to:", OUT_DIR / "training_curves_top7.png")

