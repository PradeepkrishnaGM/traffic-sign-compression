"""Slow checks, run only with RUN_SLOW=1 (they need trained checkpoints)."""
import os
import unittest

import numpy as np
import torch
from torch.nn.utils import prune

from src import config as C
from src import data, models, train, onnx_utils


@unittest.skipUnless(os.environ.get("RUN_SLOW"), "set RUN_SLOW=1 to run")
class SlowTests(unittest.TestCase):
    def test_onnx_matches_pytorch(self):
        d = data.load_arrays()
        x = d["test_x"][:128]
        model = models.make_student()
        model.load_state_dict(torch.load(C.CKPT_DIR / "student_fp32.pt"))
        a = train.predict_logits(model, x)
        sess = onnx_utils.make_session(C.CKPT_DIR / "onnx" / "student_fp32.onnx", 1)
        b = onnx_utils.ort_logits(sess, x)
        self.assertLess(np.abs(a - b).max(), 1e-3)

    def test_pruning_sparsity_matches_target(self):
        net = models.make_teacher(pretrained=False)
        params = [(m, "weight") for m in net.modules()
                  if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.7)
        zeros = sum(int((m.weight == 0).sum()) for m, _ in params)
        total = sum(m.weight.numel() for m, _ in params)
        self.assertAlmostEqual(zeros / total, 0.7, places=3)

    def test_kd_loss_equals_ce_when_alpha_zero(self):
        s, t = torch.randn(8, 43), torch.randn(8, 43)
        y = torch.randint(0, 43, (8,))
        loss = train.kd_loss(s, t, y, alpha=0.0)
        self.assertAlmostEqual(loss.item(), torch.nn.functional.cross_entropy(s, y).item(), places=5)

    def test_student_size_range(self):
        n = models.count_params(models.make_student())
        self.assertTrue(100_000 <= n <= 500_000)


if __name__ == "__main__":
    unittest.main()
