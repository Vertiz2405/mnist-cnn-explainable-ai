# CNN Architecture & Training — Report Draft

## Architecture
All configurations use Conv2D(32, 3×3, same, ReLU), MaxPooling2D(2×2), Conv2D(64, 3×3, same, ReLU), MaxPooling2D(2×2), Flatten, a variable-size Dense(ReLU) head, and Dense(10, softmax). The regularization experiment adds Dropout(0.3) before the output.

## Dataset and methodology
The original MNIST training split (60,000 images) was stratified into 50,000 training and 10,000 validation images with random_state=42. The official 10,000-image test set was held out. Images were scaled to [0,1] as float32 and reshaped to 28×28×1.
All models used Adam, sparse categorical cross-entropy, batch size 32, and up to 15 epochs with early stopping (patience 3) and best-val-loss checkpoints.

## Validation comparison
| Configuration | Dense | Dropout | LR | Parameters | Best epoch | Val accuracy | Val loss | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dropout_128 | 128 | 0.3 | 0.001 | 421,642 | 3 | 98.74% | 0.0438 | 169.5 |
| dense_64 | 64 | 0 | 0.001 | 220,234 | 2 | 98.67% | 0.0473 | 104.9 |
| dense_32 | 32 | 0 | 0.001 | 119,530 | 6 | 98.76% | 0.0476 | 173.4 |
| dense_64_lr0001 | 64 | 0 | 0.0001 | 220,234 | 14 | 98.56% | 0.0502 | 317.5 |
| baseline_128 | 128 | 0 | 0.001 | 421,642 | 2 | 98.49% | 0.0548 | 138.0 |
| dense_64_lr003 | 64 | 0 | 0.003 | 220,234 | 7 | 98.84% | 0.0587 | 217.6 |

## Selected model and test results
The model **dropout_128** was selected using minimum validation loss (0.0438, accuracy 98.74%). On the untouched test set, accuracy was **98.86%**, loss **0.0337**, macro-F1 **0.9886**, with 114 errors out of 10,000 images.

## Error analysis
Use `results/figures/test_confusion_matrix.png` and `test_misclassifications.png` to discuss recurring digit confusions and difficult handwriting. Coordinate saliency/Grad-CAM interpretation with the XAI teammate.

## Limitations
Single-seed comparisons cannot establish statistical significance. Future work: repeated seeds, data augmentation, robustness tests and probability calibration. Do not tune against the test set.

## Figures
- `results/figures/comparison_validation_accuracy.png`
- `results/figures/comparison_capacity.png`
- `results/figures/test_confusion_matrix.png`
- `results/figures/test_misclassifications.png`
- Per-experiment training curves in `results/figures/`.
