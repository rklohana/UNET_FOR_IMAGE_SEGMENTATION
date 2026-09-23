"""Smoke check for assignment U-Net dimensions when PyTorch is installed."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_valid_convolution_output_and_label_crop(self):
        import torch
        from model import UNet, center_crop

        model = UNet(base_channels=2).eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(tuple(logits.shape), (1, 2, 132, 132))
        with torch.no_grad():
            full_size_logits = model(torch.zeros(1, 1, 572, 572))
        self.assertEqual(tuple(full_size_logits.shape), (1, 2, 388, 388))
        labels = center_crop(torch.zeros(1, 320, 320), 132, 132)
        self.assertEqual(tuple(labels.shape), (1, 132, 132))


if __name__ == "__main__":
    unittest.main()
