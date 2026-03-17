"""
Verify class mapping between model and class_names.json.
This is critical - a mismatch will cause uniform-looking predictions.
"""
from pathlib import Path
import json
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
MODEL_PATH_17 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_17classes_best.keras"
MODEL_PATH_19 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_19classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

def get_model_classes(model_path: Path):
    """Get the actual classes a model was trained on by checking directory structure."""
    # The model was trained using image_dataset_from_directory
    # So we need to check what classes exist in the melspectrograms directory
    # and match them to what the model expects
    
    # For now, we'll infer from the model output shape
    model = tf.keras.models.load_model(model_path)
    num_classes = model.output_shape[1]
    return num_classes

def get_actual_class_names():
    """Get class names from directory structure."""
    species_dirs = [d.name for d in sorted(MELS_DIR.iterdir()) 
                   if d.is_dir() and d.name != "not_galago"]
    return sorted(species_dirs)

def main():
    print("=" * 70)
    print("Class Mapping Verification")
    print("=" * 70)
    
    # Check what's in class_names.json
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, 'r') as f:
            json_classes = json.load(f)
        print(f"\nclass_names.json has {len(json_classes)} classes:")
        for i, cls in enumerate(json_classes):
            print(f"  {i:2d}: {cls}")
    else:
        print(f"\nERROR: {CLASS_NAMES_PATH} not found")
        json_classes = []
    
    # Check actual directories
    actual_classes = get_actual_class_names()
    print(f"\nActual species directories: {len(actual_classes)}")
    for i, cls in enumerate(actual_classes):
        print(f"  {i:2d}: {cls}")
    
    # Check each model
    print("\n" + "=" * 70)
    print("Model Class Counts")
    print("=" * 70)
    
    models_to_check = [
        ("16-class", MODEL_PATH_16),
        ("17-class", MODEL_PATH_17),
        ("19-class", MODEL_PATH_19),
    ]
    
    for name, path in models_to_check:
        if path.exists():
            try:
                model = tf.keras.models.load_model(path)
                num_classes = model.output_shape[1]
                print(f"\n{name} model ({path.name}):")
                print(f"  Output shape: {model.output_shape}")
                print(f"  Number of classes: {num_classes}")
                
                if num_classes != len(json_classes):
                    print(f"  WARNING: Model has {num_classes} classes but class_names.json has {len(json_classes)}!")
                    print(f"  This is a CRITICAL mismatch - predictions will be wrong!")
                else:
                    print(f"  OK: Matches class_names.json ({len(json_classes)} classes)")
            except Exception as e:
                print(f"\n{name} model: ERROR loading - {e}")
        else:
            print(f"\n{name} model: Not found")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    if MODEL_PATH_16.exists():
        model_16 = tf.keras.models.load_model(MODEL_PATH_16)
        num_16 = model_16.output_shape[1]
        
        if num_16 != len(json_classes):
            print(f"\n1. FIX class_names.json to match the 16-class model")
            print(f"   The 16-class model expects {num_16} classes")
            print(f"   But class_names.json has {len(json_classes)} classes")
            print(f"   This mismatch is causing prediction errors!")
            
            # Find which 16 classes the model was trained on
            # (This would be in the training script, but we can infer from MIN_SAMPLES_PER_CLASS)
            print(f"\n2. The 16-class model likely excludes species with < 30 samples")
            print(f"   Check train_cnn_all_species.py for MIN_SAMPLES_PER_CLASS")

if __name__ == "__main__":
    main()
