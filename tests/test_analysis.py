import unittest
import numpy as np

from src import analysis


class AnalysisTests(unittest.TestCase):
    def test_flip_rate_with_itself_is_zero(self):
        pred = np.array([1, 2, 3, 4])
        self.assertEqual(analysis.flip_rate(pred, pred), 0.0)
        self.assertEqual(analysis.flip_rate(pred, np.array([1, 2, 0, 0])), 50.0)

    def test_pareto_front_ties(self):
        # same x, lower accuracy is dominated; identical points do not dominate each other
        xs, ys = [1.0, 1.0, 2.0, 3.0, 3.0], [0.90, 0.95, 0.95, 0.99, 0.99]
        self.assertEqual(analysis.pareto_front(xs, ys), [1, 3, 4])

    def test_flip_rate_parent_correct(self):
        labels = np.array([0, 1, 2, 3])
        parent = np.array([0, 1, 0, 0])          # right on the first two only
        pred = np.array([0, 2, 9, 9])            # changes image 1 (parent right), 2 and 3 (parent wrong)
        self.assertEqual(analysis.flip_rate_parent_correct(pred, parent, labels), 50.0)

    def test_spearman(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_ece_perfectly_calibrated_is_zero(self):
        # confidence 1.0 and always right
        probs = np.eye(3)[[0, 1, 2, 0]]
        self.assertAlmostEqual(analysis.ece(probs, np.array([0, 1, 2, 0])), 0.0)

    def test_ece_hand_computed(self):
        # four predictions with confidence 0.8, two right: accuracy 0.5, gap 0.3, one bin
        probs = np.array([[0.8, 0.2]] * 4)
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(analysis.ece(probs, labels), 0.3)

    def test_calibration_summary(self):
        logits = np.array([[10.0, 0.0], [10.0, 0.0]])
        s = analysis.calibration_summary(logits, np.array([0, 1]))
        self.assertAlmostEqual(s["acc"], 0.5)
        self.assertGreater(s["overconfidence"], 0.45)


if __name__ == "__main__":
    unittest.main()
