from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

IMG_HEIGHT = 128
IMG_WIDTH = 128

MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
MODEL_PATH = PROJECT_ROOT / "models" / "top6" / "galago_cnn_top6_best.keras"

# Must match SELECTED_CLASSES in train_cnn_top6.py
CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

# Load model once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_png(png_path: Path, top_k: int = 3):
    if not png_path.exists():
        print("File not found:", png_path)
        return

    print("\nLoading image:", png_path)
    img = load_img(png_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)           # 0â€“255, uint8
    x = np.expand_dims(x, axis=0)   # shape (1, H, W, 3); NO /255.0 here

    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
    top_indices = probs.argsort()[::-1][:top_k]

    print("Top predictions:")
    for idx in top_indices:
        species = CLASS_NAMES[idx]
        p = float(probs[idx])
        print(f"{species:25s}  p = {p:.3f}")

if __name__ == "__main__":
    # Change this line each time to test a different PNG
    example_png = MELS_DIR / "Galagoides_sp_nov" / "Incemental call  G sp nov 1_aug1.png"
    predict_png(example_png)

