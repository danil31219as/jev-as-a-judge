import importlib.util
import unittest


class JudgeRlTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_bfloat16_assessor_probabilities_are_renormalized(self):
        import torch
        from judge_model import normalize_soft_targets

        labels = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0]], dtype=torch.bfloat16)
        valid = torch.tensor([[True, True, True, False]])
        normalized = normalize_soft_targets(labels, valid)
        self.assertAlmostEqual(normalized.sum().item(), 1.0, places=6)
        self.assertEqual(normalized[0, 3].item(), 0.0)
        with self.assertRaisesRegex(ValueError, "got sums"):
            normalize_soft_targets(torch.tensor([[0.2, 0.2, 0.2, 0.0]]), valid)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_rlcd_has_finite_gradient_and_ignores_padding(self):
        import torch
        from judge_model import laya_rlcd_loss

        torch.manual_seed(42)
        logits = torch.tensor([[0.2, -0.1, -1e4]], requires_grad=True)
        targets = torch.tensor([[0.25, 0.75, 0.0]])
        valid = torch.tensor([[True, True, False]])
        values = torch.tensor([[1, 0, -1]])
        loss = laya_rlcd_loss(
            logits, targets, valid, values, sigma=0.4, group_size=4
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertEqual(logits.grad[0, 2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
