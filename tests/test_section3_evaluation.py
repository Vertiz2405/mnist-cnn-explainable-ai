"""Checks for repeated preprocessing, invalid predictions and reference mismatches."""

from pathlib import Path
import json
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.section3_evaluation import (
    preprocess_raw_images, evaluate_predictions, check_reference,
)


class EvaluationChecks(unittest.TestCase):
    def test_raw_pixels_are_scaled_without_changing_the_source(self):
        raw = np.zeros((2, 28, 28), dtype=np.uint8)
        raw[0, 0, :3] = [0, 128, 255]
        original = raw.copy()
        prepared = preprocess_raw_images(raw)
        self.assertEqual(prepared.shape, (2, 28, 28, 1))
        self.assertEqual(prepared.dtype, np.float32)
        np.testing.assert_allclose(prepared[0, 0, :3, 0], [0, 128 / 255, 1])
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(preprocess_raw_images(raw), prepared)

    def test_already_prepared_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_raw_images(np.zeros((2, 28, 28, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess_raw_images(np.zeros((2, 28, 28), dtype=np.float32))

    def test_perfect_predictions_have_no_ranked_confusions(self):
        labels = np.arange(10)
        _, report, matrix, wrong, ranked, summary = evaluate_predictions(labels, np.eye(10))
        self.assertEqual(summary["test_accuracy"], 1.0)
        self.assertEqual(summary["incorrect_predictions"], 0)
        self.assertEqual(report["macro avg"]["f1-score"], 1.0)
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(len(wrong), 0)
        self.assertTrue(ranked.empty)

    def test_malformed_probabilities_are_rejected(self):
        for probabilities in [np.ones((2, 9)), np.ones((2, 10)), np.full((2, 10), np.nan)]:
            with self.assertRaises(ValueError):
                evaluate_predictions(np.array([0, 1]), probabilities)

    def test_matching_accuracy_cannot_hide_a_different_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = np.eye(10, dtype=int)
            (root / "test_metrics.json").write_text(json.dumps({"test_accuracy": 1.0}))
            pd.DataFrame(matrix).to_csv(root / "test_confusion_matrix.csv")
            checks = check_reference({"test_accuracy": 1.0}, matrix, root)
            self.assertTrue(all(checks.values()))
            altered = matrix.copy()
            altered[4, 9] = 1
            with self.assertRaises(ValueError):
                check_reference({"test_accuracy": 1.0}, altered, root)


if __name__ == "__main__":
    unittest.main()
