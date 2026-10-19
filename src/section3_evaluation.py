"""Prepare MNIST and independently evaluate the group's selected CNN.

Run from the repository root with ``python -m src.section3_evaluation``.
The selected model is restored, never trained. Reference results are read only.
"""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


CLASSES = list(range(10))
ROOT = Path(__file__).resolve().parents[1]


def preprocess_raw_images(images):
    """Convert raw uint8 MNIST images to float32 arrays of shape (N, 28, 28, 1)."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Expected a non-empty raw image array with shape (N, 28, 28).")
    if images.dtype != np.uint8:
        raise ValueError("Expected raw uint8 pixels; do not pass already-scaled images.")
    return (images.astype(np.float32) / 255.0)[..., np.newaxis]


def split_training_data(images, labels):
    """Recreate the 50,000/10,000 stratified split used during training."""
    return train_test_split(
        images, labels, test_size=10_000, random_state=42,
        shuffle=True, stratify=labels,
    )


def load_selected_model(model_path):
    """Restore the exact selected architecture and its previously trained weights.

    Loading weights avoids initializer-configuration differences between Keras
    versions. The architecture matches the selected dropout_128 model in notebook 03.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model not found: {model_path}")
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.load_weights(model_path)
    if model.input_shape != (None, 28, 28, 1) or model.output_shape != (None, 10):
        raise ValueError("The restored model has unexpected input or output dimensions.")
    if model.count_params() != 421_642:
        raise ValueError("The restored architecture does not match the selected model.")
    return model


def evaluate_predictions(labels, probabilities):
    """Calculate classification scores and rank actual/predicted digit confusions."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a non-empty one-dimensional integer label array.")
    if not np.isin(labels, CLASSES).all():
        raise ValueError("Labels must be digits from 0 to 9.")
    if probabilities.shape != (len(labels), 10):
        raise ValueError("Expected one row of ten probabilities per test image.")
    if (not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Predictions must be finite probabilities summing to one.")
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        labels, predictions, labels=CLASSES, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=CLASSES)
    wrong_indices = np.flatnonzero(predictions != labels)
    summary = {
        "test_images": int(len(labels)),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "correct_predictions": int(np.trace(matrix)),
        "incorrect_predictions": int(len(wrong_indices)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    rows = [
        {"Actual digit": actual, "Predicted digit": predicted,
         "Errors": int(matrix[actual, predicted])}
        for actual in CLASSES for predicted in CLASSES
        if actual != predicted and matrix[actual, predicted] > 0
    ]
    ranked = pd.DataFrame(rows, columns=["Actual digit", "Predicted digit", "Errors"])
    ranked = ranked.sort_values(
        ["Errors", "Actual digit", "Predicted digit"],
        ascending=[False, True, True], ignore_index=True,
    )
    return predictions, report, matrix, wrong_indices, ranked, summary


def check_reference(summary, matrix, reference_dir):
    """Require matching accuracy and all 100 reference confusion-matrix counts."""
    reference_dir = Path(reference_dir)
    reference = json.loads((reference_dir / "test_metrics.json").read_text())
    reference_matrix = pd.read_csv(
        reference_dir / "test_confusion_matrix.csv", index_col=0,
    ).to_numpy()
    checks = {
        "accuracy_matches_reference": bool(np.isclose(
            summary["test_accuracy"], reference["test_accuracy"], atol=1e-6, rtol=0,
        )),
        "confusion_matrix_matches_reference": bool(np.array_equal(matrix, reference_matrix)),
    }
    if not all(checks.values()):
        raise ValueError(f"Evaluation does not match the supplied reference: {checks}")
    return checks


def make_confusion_figure(matrix):
    """Plot rows as actual digits and columns as predictions."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
        ax=ax, cmap="Greys", values_format="d", colorbar=False,
    )
    ax.set(title="MNIST test confusion matrix", xlabel="Predicted digit", ylabel="Actual digit")
    fig.tight_layout()
    return fig


def make_error_figure(images, labels, predictions, probabilities, wrong_indices):
    """Show the first 12 errors in test order, not a representative random sample."""
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, wrong_indices[:12]):
        predicted = predictions[index]
        ax.imshow(images[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"Actual {labels[index]} | Predicted {predicted}\n"
            f"Output probability: {probabilities[index, predicted]:.1%}"
        )
    if len(wrong_indices) == 0:
        fig.suptitle("No misclassified test images")
    fig.tight_layout()
    return fig


def save_results(output_dir, report, matrix, ranked, summary, confusion_fig, error_fig):
    """Write independent Section 3 outputs without overwriting training references."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(report).T.drop(index="accuracy")
    table.to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output_dir / "confusion_matrix.csv")
    ranked.to_csv(output_dir / "ranked_confusions.csv", index=False)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    confusion_fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    error_fig.savefig(output_dir / "misclassified_examples.png", dpi=180)


def checkpoint_sha256(path):
    """Identify the exact checkpoint used for evaluation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import tensorflow as tf

    model_path = ROOT / "results/models/cnn_selected_for_xai.keras"
    model = load_selected_model(model_path)
    (full_raw, full_labels), (test_raw, test_labels) = tf.keras.datasets.mnist.load_data()
    train_raw, validation_raw, _, _ = split_training_data(full_raw, full_labels)
    for name, raw in [("Training", train_raw), ("Validation", validation_raw), ("Test", test_raw)]:
        prepared = preprocess_raw_images(raw)
        print(name, prepared.shape, prepared.dtype, float(prepared.min()), float(prepared.max()))
    test_images = preprocess_raw_images(test_raw)
    probabilities = model.predict(test_images, batch_size=256, verbose=0)
    predictions, report, matrix, wrong, ranked, summary = evaluate_predictions(test_labels, probabilities)
    summary.update(check_reference(summary, matrix, ROOT / "results/metrics"))
    summary["checkpoint_sha256"] = checkpoint_sha256(model_path)
    summary["model_parameters"] = int(model.count_params())
    summary["tensorflow_version"] = tf.__version__
    confusion_fig = make_confusion_figure(matrix)
    error_fig = make_error_figure(test_images, test_labels, predictions, probabilities, wrong)
    output_dir = ROOT / "results/section3"
    save_results(output_dir, report, matrix, ranked, summary, confusion_fig, error_fig)
    plt.close(confusion_fig)
    plt.close(error_fig)
    print(json.dumps(summary, indent=2))
    print("Results saved in results/section3/")


if __name__ == "__main__":
    main()
