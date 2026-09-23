"""Checks for BBBC019 discovery and source-stratified splitting."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bbbc019 import (discover_bbbc019_pairs, make_source_holdout_split,
                     make_source_stratified_split)
from preprocessing import Cell_data, load_split


class BBBC019Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in ("Init", "SN15", "HEK293", "MDCK"):
            image_dir = self.root / source / "images"
            mask_dir = self.root / source / "manual"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for index in range(6):
                image = np.full((20, 32), 100 + index, dtype=np.uint8)
                mask = np.zeros((20, 32), dtype=np.uint8)
                mask[4:16, 8:24] = 255
                Image.fromarray(image).save(image_dir / f"sample_{index}.tif")
                Image.fromarray(mask).save(mask_dir / f"sample_{index}_manual.png")

    def test_discovery_split_and_path_aware_loading(self):
        pairs = discover_bbbc019_pairs(self.root)
        self.assertEqual(len(pairs), 24)
        split = make_source_stratified_split(pairs, seed=9)
        for name in ("train", "val", "test"):
            self.assertEqual(
                {pair["source"] for pair in split[name]},
                {"Init", "SN15", "HEK293", "MDCK"},
            )
        split_path = self.root / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        self.assertEqual(load_split(split_path, self.root), split)
        image, mask = Cell_data(self.root, split["train"], size=320)[0]
        self.assertEqual(image.shape, (1, 320, 320))
        self.assertEqual(mask.shape, (320, 320))
        self.assertTrue(np.all(mask[:, :60] == 0))

    def test_source_holdout_has_no_source_overlap(self):
        pairs = discover_bbbc019_pairs(self.root)
        split = make_source_holdout_split(
            pairs, val_sources=("SN15",), test_sources=("HEK293", "MDCK")
        )
        sources = [{pair["source"] for pair in split[name]}
                   for name in ("train", "val", "test")]
        self.assertEqual(sources[0], {"Init"})
        self.assertEqual(sources[1], {"SN15"})
        self.assertEqual(sources[2], {"HEK293", "MDCK"})
        self.assertFalse(sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2])

    def test_missing_manual_mask_is_rejected(self):
        (self.root / "SN15" / "manual" / "sample_0_manual.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing manual mask"):
            discover_bbbc019_pairs(self.root)


if __name__ == "__main__":
    unittest.main()
