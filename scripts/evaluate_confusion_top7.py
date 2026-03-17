from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

IMG_HEIGHT = 128
IMG_WIDTH = 128

MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
MODEL_PATH = PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"

# Must match SELECTED_CLASSES in train_cnn_top7.py
SELECTED_CLASSES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Paragalago_zanzibaricus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

model = tf.keras.models.load_model(MODEL_PATH)

val_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=0.2,
    subset="validation",
    seed=42,
    class_names=SELECTED_CLASSES,
)

class_names = val_ds.class_names
print("Classes (top7):", class_names)

y_true = []
y_pred = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification report (top7):\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix â€“ Top 7 Species")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

out_path = MODEL_PATH.parent / "confusion_matrix_top7.png"
plt.savefig(out_path, dpi=150)
plt.close()

print("Saved confusion matrix to:", out_path)

