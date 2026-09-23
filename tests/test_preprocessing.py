"""Small fixture checks for splitting and paired image preprocessing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import CellDataset, discover_pairs, load_split, make_split


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scans").mkdir()
        (self.root / "labels").mkdir()
        for index in range(5):
            image = np.zeros((32, 32), dtype=np.uint8)
            image[8:24, 8:24] = 255
            mask = image.copy()
            Image.fromarray(image).save(self.root / "scans" / f"cell_{index}.png")
            Image.fromarray(mask).save(self.root / "labels" / f"cell_{index}.bmp")

    def test_split_is_reproducible_and_disjoint(self):
        pairs = discover_pairs(self.root)
        first = make_split(pairs, seed=7)
        self.assertEqual(first, make_split(pairs, seed=7))
        self.assertEqual(sum(len(first[name]) for name in ("train", "val", "test")), 5)
        import json
        split_path = self.root / "split.json"
        split_path.write_text(json.dumps(first), encoding="utf-8")
        self.assertEqual(load_split(split_path, self.root), first)

    def test_scaling_and_mask_shape(self):
        dataset = CellDataset(self.root, discover_pairs(self.root), size=320)
        image, mask = dataset[0]
        self.assertEqual(image.shape, (1, 320, 320))
        self.assertEqual(mask.shape, (320, 320))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(mask.dtype, np.int64)
        self.assertEqual(set(np.unique(mask)), {0, 1})
        self.assertEqual(float(image.max()), 1.0)

    def test_augmented_masks_remain_binary(self):
        dataset = CellDataset(self.root, discover_pairs(self.root), size=320, augment=True)
        for _ in range(5):
            image, mask = dataset[0]
            self.assertEqual(image.shape, (1, 320, 320))
            self.assertEqual(set(np.unique(mask)), {0, 1})
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_missing_pair_is_rejected(self):
        (self.root / "labels" / "cell_0.bmp").unlink()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            discover_pairs(self.root)


if __name__ == "__main__":
    unittest.main()
