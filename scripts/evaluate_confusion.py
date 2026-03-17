from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

IMG_HEIGHT = 128
IMG_WIDTH = 128

MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
MODEL_PATH = Path(r"C:\Users\galag\GitHub\galago_audio_project\models\galago_cnn_best.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# Load validation set the same way as training
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
)

class_names = val_ds.class_names
print("Classes:", class_names)

# Collect predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Determine which classes are present in validation
unique_labels = sorted(np.unique(y_true))
print("\nLabels present in validation set:", unique_labels)

# Filter class names accordingly
present_class_names = [class_names[i] for i in unique_labels]

print("\nClassification report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=unique_labels,
    target_names=present_class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(present_class_names))
plt.xticks(tick_marks, present_class_names, rotation=90)
plt.yticks(tick_marks, present_class_names)

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

out_path = MODEL_PATH.parent / "confusion_matrix.png"
plt.savefig(out_path, dpi=150)
plt.close()

print("Saved confusion matrix to:", out_path)

