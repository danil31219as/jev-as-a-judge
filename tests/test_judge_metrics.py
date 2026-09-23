import math
import unittest
from types import SimpleNamespace

from judge_metrics import compute_judge_metrics, score_metrics


class JudgeMetricTests(unittest.TestCase):
    def test_expected_score_errors_and_tied_assessor_mode(self):
        predictions = [[math.log(3), 0.0, -1e4], [0.0, math.log(3), -1e4]]
        targets = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]
        metrics = compute_judge_metrics(SimpleNamespace(
            predictions=predictions, label_ids=targets
        ))
        self.assertAlmostEqual(metrics["mae"], 0.25)
        self.assertAlmostEqual(metrics["rmse"], 0.25)
        self.assertEqual(metrics["f1_macro"], 1.0)

    def test_macro_f1_counts_all_observed_classes(self):
        metrics = score_metrics([[0.0, 2.0], [2.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
