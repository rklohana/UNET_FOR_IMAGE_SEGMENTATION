"""The generated portfolio report includes the assignment's run details."""

import tempfile
import unittest
from pathlib import Path

from reporting import save_loss_plot, write_report


class ReportingTests(unittest.TestCase):
    def test_plot_and_report(self):
        history = [
            {"epoch": 1, "train_loss": 0.8, "loss": 0.9},
            {"epoch": 2, "train_loss": 0.5, "loss": 0.6},
        ]
        summary = {
            "image_size": 572, "output_height": 388, "output_width": 388,
            "base_channels": 64, "train_samples": 30, "val_samples": 4,
            "test_samples": 4, "epochs": 2, "best_epoch": 2, "batch_size": 1,
            "learning_rate": 0.001, "device": "cpu", "training_seconds": 12.5,
        }
        metrics = {"loss": 0.55, "pixel_accuracy": 0.9, "dice": 0.8,
                   "iou": 0.67, "precision": 0.81, "recall": 0.79,
                   "samples": 4}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_loss_plot(history, output / "loss_curves.png", metrics["loss"])
            write_report(summary, metrics, ["cell_preview.png"], output)
            report = (output / "README.md").read_text(encoding="utf-8")
            self.assertGreater((output / "loss_curves.png").stat().st_size, 0)
            for value in ("572 x 572", "388 x 388", "12.5 seconds", "0.8000",
                          "cell_preview.png", "Held out test"):
                self.assertIn(value, report)


if __name__ == "__main__":
    unittest.main()
