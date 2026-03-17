"""\
Fine-tune the existing 16-class classifier on `data/raw_audio` windows.

Goal: reduce domain shift between training (PNG dataset) and your evaluation WAVs.

- Uses the same windowing + mel->RGB pipeline as `predict_3stage_with_context.py`
- Splits by WAV file into train/val
- Saves a new model: `models/all_species/galago_cnn_all_16classes_finetuned_raw_audio.keras`

NOTE: This is *domain adaptation* on a small labeled set (69 files). It can overfit.
"""

from __future__ import annotations

from pathlib import Path
import json
import random
from collections import Counter

import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
MODEL_IN = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
MODEL_OUT = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_finetuned_raw_audio.keras"

# Match predict_3stage_with_context
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128
WINDOW_SEC = 2.5
HOP_SEC = 1.25
MIN_WINDOWS = 1

# Fine-tune params
SEED = 42
VAL_SPLIT_FILES = 0.2
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-5

LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
}


def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    n_mels, T = S.shape
    if T == target_frames:
        return S
    if T > target_frames:
        start = (T - target_frames) // 2
        return S[:, start : start + target_frames]
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    return np.pad(S, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=pad_value)


def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts


def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_padded = pad_or_crop(S_db, TARGET_FRAMES)
    S_min = S_padded.min()
    S_max = S_padded.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_padded)
    else:
        S_norm = (S_padded - S_min) / (S_max - S_min)
    cmap = plt.colormaps["magma"]
    S_rgb = cmap(S_norm)[:, :, :3]
    S_rgb_float = (S_rgb * 255.0).astype(np.float32)
    return S_rgb_float


def label_from_path(wav_path: Path) -> str | None:
    # folder under raw_audio
    try:
        rel = wav_path.relative_to(AUDIO_DIR)
        folder = rel.parts[0]
    except Exception:
        folder = wav_path.parent.name
    return LABEL_MAP.get(folder, folder)


def build_examples(wav_paths: list[Path], class_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for wav in wav_paths:
        lbl = label_from_path(wav)
        if lbl not in class_to_idx:
            continue
        try:
            y, sr = librosa.load(str(wav), sr=SR, mono=True)
        except Exception:
            continue

        starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
        for start in starts:
            end = min(len(y), start + int(WINDOW_SEC * sr))
            y_win = y[start:end]
            if len(y_win) < int(0.5 * sr):
                continue
            rgb = wav_window_to_rgb_fixed(y_win, sr)
            X_list.append(rgb)
            y_list.append(class_to_idx[lbl])

    if not X_list:
        return np.empty((0, 128, 128, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def main():
    print("Fine-tuning on raw_audio windows")

    if not MODEL_IN.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_IN}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing class list: {CLASS_NAMES_PATH}")

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    wavs = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No wav files under {AUDIO_DIR}")

    # Split by file
    random.seed(SEED)
    wavs_shuf = wavs[:]
    random.shuffle(wavs_shuf)
    split = int(len(wavs_shuf) * (1 - VAL_SPLIT_FILES))
    train_files = wavs_shuf[:split]
    val_files = wavs_shuf[split:]

    print(f"WAV files: {len(wavs)} (train {len(train_files)}, val {len(val_files)})")

    print("Building window datasets (this can take a minute)...")
    X_train, y_train = build_examples(train_files, class_to_idx)
    X_val, y_val = build_examples(val_files, class_to_idx)

    print(f"Windows: train {len(X_train)}, val {len(X_val)}")
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("No training/validation windows built; check labels/mapping.")

    # Class weights on windows
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    class_weights = {i: total / (len(class_names) * counts.get(i, 1)) for i in range(len(class_names))}

    # Build tf.data
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Loading base model and fine-tuning...")
    model = tf.keras.models.load_model(MODEL_IN)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    hist = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(ds_val, verbose=0)
    print(f"Val accuracy (raw_audio file-split): {val_acc:.3f}")

    model.save(MODEL_OUT)
    print(f"Saved: {MODEL_OUT}")

    # Save training metadata
    meta = {
        "base_model": MODEL_IN.name,
        "fine_tuned_on": "data/raw_audio",
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "val_split_files": VAL_SPLIT_FILES,
        "epochs_ran": len(hist.history.get("loss", [])),
        "val_acc": float(val_acc),
    }
    (MODEL_OUT.parent / "finetune_raw_audio_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
