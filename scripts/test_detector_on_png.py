"""
Test detector directly on a PNG mel-spectrogram from training data.
This will verify if the model works at all.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "galago_detector_best.keras"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

def main():
    print("Testing Detector on Training PNG Files")
    print("="*60)
    
    detector = tf.keras.models.load_model(DETECTOR_PATH)
    
    # Find some PNG files from training
    galago_pngs = list((MELS_DIR / "Galago_senegalensis").glob("*.png"))[:5]
    not_galago_pngs = list((MELS_DIR / "not_galago").glob("*.png"))[:5]
    
    print(f"\nTesting on {len(galago_pngs)} galago PNGs and {len(not_galago_pngs)} not_galago PNGs...")
    
    print(f"\n{'File':<50} {'Prediction':<15} {'Probability':<12}")
    print("-"*80)
    
    # Test galago PNGs
    for png in galago_pngs:
        img = Image.open(png).convert('RGB')
        img = img.resize((128, 128))  # Resize to match model input
        img_array = np.array(img).astype(np.float32)  # [0, 255] float32
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        prob = detector.predict(img_array, verbose=0)[0][0]
        decision = "GALAGO" if prob > 0.5 else "NOT GALAGO"
        print(f"{png.name[:49]:<50} {decision:<15} {prob:.4f}")
    
    print()
    
    # Test not_galago PNGs
    for png in not_galago_pngs:
        img = Image.open(png).convert('RGB')
        img = img.resize((128, 128))  # Resize to match model input
        img_array = np.array(img).astype(np.float32)  # [0, 255] float32
        img_array = np.expand_dims(img_array, axis=0)
        
        prob = detector.predict(img_array, verbose=0)[0][0]
        decision = "GALAGO" if prob > 0.5 else "NOT GALAGO"
        print(f"{png.name[:49]:<50} {decision:<15} {prob:.4f}")

if __name__ == "__main__":
    main()

