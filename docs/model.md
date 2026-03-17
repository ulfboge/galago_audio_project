# Model architecture and training

## Architecture

The classifier is a convolutional neural network (CNN) composed of:

- Four convolutional blocks with increasing filter depth
- Max-pooling layers for spatial reduction
- Dropout layers to reduce overfitting
- A dense classification head with softmax output

The final layer has **7 output neurons**, one per species.

---

## Training strategy

- Input: 128 × 128 RGB mel-spectrograms
- Loss function: categorical cross-entropy
- Optimizer: Adam
- Class imbalance handled using class weighting

---

## Incremental species expansion

Species were added **one at a time**, with evaluation after each addition:

1. Initial 6-species model
2. Addition of *Paragalago zanzibaricus* (top-7)

This incremental approach ensures that new classes improve model behavior without degrading existing performance.

---

## Model selection

The best model checkpoint is selected based on **validation loss**, not final epoch performance.

Recommended model file: models_top7/galago_cnn_top7_best.keras
