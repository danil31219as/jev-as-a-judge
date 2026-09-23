import math
import unittest
from types import SimpleNamespace

from judge_metrics import compute_judge_metrics, score_metrics


class JudgeMetricTests(unittest.TestCase):
    def test_tied_assessor_votes_use_lower_score(self):
        predictions = [[0.0, 0.0, -1e4], [0.0, math.log(3), -1e4]]
        targets = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]
        metrics = compute_judge_metrics(SimpleNamespace(
            predictions=predictions, label_ids=targets
        ))
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["f1_macro"], 1.0)

    def test_mae_and_rmse_compare_argmax_to_majority(self):
        metrics = score_metrics(
            [[2.0, 1.0, 0.0], [0.0, 0.1, 3.0]],
            [[0.1, 0.8, 0.1], [0.6, 0.4, 0.0]],
        )
        self.assertEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.5))
        self.assertEqual(metrics["f1_macro"], 0.0)

    def test_macro_f1_counts_all_observed_classes(self):
        metrics = score_metrics([[0.0, 2.0], [2.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
