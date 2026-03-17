"""
Sanity check: Overfit a tiny subset to verify the model can learn.
This should achieve ~100% training accuracy within minutes.
If it can't overfit, there's a bug in labels, loss, model, or preprocessing.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "models" / "sanity_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 4
EPOCHS = 20

def load_image_and_preprocess(file_path, label):
    """Load and preprocess image."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def main():
    print("=" * 70)
    print("Sanity Check: Overfit Tiny Subset")
    print("=" * 70)
    print("\nThis test should achieve ~100% training accuracy quickly.")
    print("If it can't, there's a bug in the training setup.\n")
    
    # Find a few species with samples
    species_dirs = [d for d in sorted(MELS_DIR.iterdir()) 
                   if d.is_dir() and d.name != "not_galago"]
    
    if len(species_dirs) < 2:
        print("ERROR: Need at least 2 species for this test")
        return
    
    # Use first 2 species, 5 samples each
    selected_species = []
    all_files = []
    all_labels = []
    
    for i, species_dir in enumerate(species_dirs[:2]):
        png_files = list(species_dir.glob("*.png"))
        if len(png_files) == 0:
            continue
        
        selected_species.append(species_dir.name)
        # Take first 5 samples
        for png_file in png_files[:5]:
            all_files.append(str(png_file))
            all_labels.append(i)
    
    if len(selected_species) < 2:
        print("ERROR: Need at least 2 species with samples")
        return
    
    print(f"Selected species:")
    for i, species in enumerate(selected_species):
        count = sum(1 for l in all_labels if l == i)
        print(f"  {i}: {species} ({count} samples)")
    
    print(f"\nTotal samples: {len(all_files)}")
    print(f"Classes: {len(selected_species)}")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    dataset = dataset.map(load_image_and_preprocess)
    dataset = dataset.batch(BATCH_SIZE)
    
    # Build simple model
    print("\nBuilding model...")
    model = keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # NOTE: inputs are already scaled to [0,1] in load_image_and_preprocess()
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(selected_species), activation="softmax"),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.summary()
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    print("Expected: Training accuracy should reach ~100% quickly\n")
    
    history = model.fit(
        dataset,
        epochs=EPOCHS,
        verbose=1,
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    final_acc = history.history['accuracy'][-1]
    final_loss = history.history['loss'][-1]
    
    print(f"\nFinal training accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"Final training loss: {final_loss:.4f}")
    
    if final_acc > 0.95:
        print("\n[PASS] Model can overfit - training setup is correct")
        print("  The issue is likely in:")
        print("    - Data distribution/imbalance")
        print("    - Model capacity for 19 classes")
        print("    - Preprocessing mismatch between train/inference")
    else:
        print("\n[FAIL] Model cannot overfit - there's a bug!")
        print("  Check:")
        print("    - Label mapping (class indices)")
        print("    - Loss function")
        print("    - Model architecture")
        print("    - Preprocessing pipeline")
    
    # Test on same data
    print("\nTesting on training data...")
    test_results = model.evaluate(dataset, verbose=0)
    test_acc = test_results[1]
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # Save model for inspection
    model_path = OUT_DIR / "sanity_check_model.keras"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
