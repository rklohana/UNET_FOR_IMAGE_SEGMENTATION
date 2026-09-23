"""Validate paired cell images, create a reproducible split, and load samples."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from torch.utils.data import Dataset
except ImportError:  # Pair validation can run before PyTorch is installed.
    Dataset = object


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _files_by_stem(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    files = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in files:
                raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
            files[key] = path
    return files


def discover_pairs(data_dir):
    data_dir = Path(data_dir)
    scans = _files_by_stem(data_dir / "scans")
    labels = _files_by_stem(data_dir / "labels")
    if not scans:
        raise ValueError(f"No scan images found in {data_dir / 'scans'}")
    if scans.keys() != labels.keys():
        missing_labels = sorted(scans.keys() - labels.keys())
        missing_scans = sorted(labels.keys() - scans.keys())
        raise ValueError(
            f"Unpaired files: missing labels for {missing_labels}; "
            f"missing scans for {missing_scans}"
        )
    pairs = []
    for key in sorted(scans):
        scan, label = scans[key], labels[key]
        with Image.open(scan) as image, Image.open(label) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image and mask sizes differ: {scan.name}, {label.name}")
            image.verify()
            mask.verify()
        pairs.append({"scan": scan.name, "label": label.name})
    return pairs


def make_split(pairs, train_fraction=0.8, val_fraction=0.1, seed=42):
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than 1")
    if len(pairs) < 3:
        raise ValueError("At least three image/mask pairs are needed")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_count = min(len(shuffled) - 2, max(1, round(len(shuffled) * train_fraction)))
    val_count = min(len(shuffled) - train_count - 1, max(1, round(len(shuffled) * val_fraction)))
    return {
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "train": sorted(shuffled[:train_count], key=lambda pair: pair["scan"]),
        "val": sorted(shuffled[train_count:train_count + val_count], key=lambda pair: pair["scan"]),
        "test": sorted(shuffled[train_count + val_count:], key=lambda pair: pair["scan"]),
    }


def load_split(path, data_dir):
    with open(path, encoding="utf-8") as handle:
        split = json.load(handle)
    if split.get("dataset") == "BBBC019v2":
        from bbbc019 import discover_bbbc019_pairs
        discovered = discover_bbbc019_pairs(data_dir)
    else:
        discovered = discover_pairs(data_dir)
    current = {(pair["scan"], pair["label"]) for pair in discovered}
    train = {(pair["scan"], pair["label"]) for pair in split["train"]}
    val = {(pair["scan"], pair["label"]) for pair in split["val"]}
    test = {(pair["scan"], pair["label"]) for pair in split["test"]}
    if not train or not val or not test or train & val or train & test or val & test or train | val | test != current:
        raise ValueError("Split must contain every pair exactly once, with nonempty train, val, and test sets")
    if any(len(group) != len(split[name]) for name, group in (("train", train), ("val", val), ("test", test))):
        raise ValueError("Split contains duplicate pairs")
    return split


def _resolve_pair_path(data_dir, value, default_directory):
    """Resolve new relative paths and legacy filename-only split entries."""
    relative = Path(value)
    direct = data_dir / relative
    if direct.is_file():
        return direct
    legacy = data_dir / default_directory / relative
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"Could not resolve {value} below {data_dir}")


def _pad_pair_to_square(image, mask):
    """Preserve aspect ratio by padding before square model resizing."""
    side = max(image.size)
    if image.size == (side, side):
        return image, mask
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    fill = int(np.median(np.asarray(image)))
    padded_image = Image.new("L", (side, side), color=fill)
    padded_mask = Image.new("L", (side, side), color=0)
    padded_image.paste(image, (left, top))
    padded_mask.paste(mask, (left, top))
    return padded_image, padded_mask


class Cell_data(Dataset):
    """PyTorch DataLoader compatible dataset; masks contain class IDs 0 or 1."""

    def __init__(self, data_dir, pairs, size=572, augment=False):
        if size < 320:
            raise ValueError("size must be at least 320 to produce a 128x128 or larger mask")
        self.data_dir = Path(data_dir)
        self.pairs = list(pairs)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        scan_path = _resolve_pair_path(self.data_dir, pair["scan"], "scans")
        label_path = _resolve_pair_path(self.data_dir, pair["label"], "labels")
        with Image.open(scan_path) as source:
            image = source.convert("L")
        with Image.open(label_path) as source:
            mask = source.convert("L")
        image, mask = _pad_pair_to_square(image, mask)
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            turns = random.randrange(4)
            if turns:
                angle = 90 * turns
                image, mask = image.rotate(angle), mask.rotate(angle)

            # Zoom in with the same crop on both images; masks stay categorical.
            if random.random() < 0.5:
                crop_size = round(self.size / random.uniform(1.05, 1.25))
                left = random.randrange(self.size - crop_size + 1)
                top = random.randrange(self.size - crop_size + 1)
                box = (left, top, left + crop_size, top + crop_size)
                image = image.crop(box).resize((self.size, self.size), Image.Resampling.BILINEAR)
                mask = mask.crop(box).resize((self.size, self.size), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            image_array = np.power(image_array, random.uniform(0.7, 1.4)).astype(np.float32)
        image_array = image_array[None, :, :]
        mask_array = (np.asarray(mask) > 0).astype(np.int64)
        return image_array, mask_array


CellDataset = Cell_data  # Backward-compatible name for existing scripts.


def main():
    parser = argparse.ArgumentParser(description="Validate scans and labels and save a train/val/test split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("split.json"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split = make_split(discover_pairs(args.data_dir), args.train_fraction, args.val_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(split['train'])} training, {len(split['val'])} validation, "
          f"and {len(split['test'])} test pairs to {args.output}")


if __name__ == "__main__":
    main()
