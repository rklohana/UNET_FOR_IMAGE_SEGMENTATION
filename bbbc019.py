"""Download, validate, and split the official BBBC019v2 dataset."""

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC019"
ARCHIVES = {
    "TScratch": "TScratch.zip",
    "Melanoma": "Melanoma.zip",
    "Init": "Init.zip",
    "SN15": "SN15.zip",
    "Scatter": "Scatter.zip",
    "Microfluidics": "Microfluidic.zip",
    "HEK293": "HEK293.zip",
    "MDCK": "MDCK.zip",
}
EXPECTED_COUNTS = {
    "TScratch": 24,
    "Melanoma": 20,
    "Init": 28,
    "SN15": 54,
    "Scatter": 6,
    "Microfluidics": 13,
    "HEK293": 12,
    "MDCK": 14,
}
# The BBBC page catalogs 28 Init images, while the archive used by the
# completed Colab run exposed 22 valid image-mask pairs. Accept both layouts
# and report the discovered count in the saved split.
ALLOWED_PAIRED_COUNTS = {
    source: ({22, 28} if source == "Init" else {count})
    for source, count in EXPECTED_COUNTS.items()
}
SOURCE_ALIASES = {"Microfluidic": "Microfluidics"}
DEFAULT_VAL_SOURCES = ("Microfluidics", "Scatter")
DEFAULT_TEST_SOURCES = ("HEK293", "MDCK")
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe path in {archive_path}: {member.filename}")
        archive.extractall(destination)


def download_bbbc019(data_dir, archive_dir=None):
    """Download and extract the eight official BBBC019v2 archives."""
    data_dir = Path(data_dir)
    archive_dir = Path(archive_dir or data_dir.parent / "BBBC019_archives")
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in ARCHIVES.items():
        archive_path = archive_dir / filename
        if not archive_path.exists():
            url = f"{BASE_URL}/{filename}"
            print(f"Downloading {source}: {url}")
            urllib.request.urlretrieve(url, archive_path)
        existing = list(data_dir.rglob(f"{source}/images"))
        if source == "Microfluidics":
            existing += list(data_dir.rglob("Microfluidic/images"))
        if not existing:
            print(f"Extracting {archive_path.name}")
            _safe_extract(archive_path, data_dir)
    pairs = discover_bbbc019_pairs(data_dir)
    counts = Counter(pair["source"] for pair in pairs)
    unexpected = {
        source: counts.get(source, 0)
        for source, allowed in ALLOWED_PAIRED_COUNTS.items()
        if counts.get(source, 0) not in allowed
    }
    extra_sources = sorted(set(counts) - set(ALLOWED_PAIRED_COUNTS))
    if unexpected or extra_sources:
        raise ValueError(
            "Unexpected BBBC019 pair counts. "
            f"Allowed {ALLOWED_PAIRED_COUNTS}; found {dict(counts)}; "
            f"extra sources {extra_sources}"
        )
    print(f"BBBC019 ready: {len(pairs)} paired images in {data_dir}")
    return pairs


def discover_bbbc019_pairs(data_dir):
    """Find official image/manual-mask pairs after archive extraction."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing BBBC019 directory: {data_dir}")
    pairs = []
    for image_dir in sorted(data_dir.rglob("images")):
        if "__MACOSX" in image_dir.parts:
            continue
        manual_dir = image_dir.parent / "manual"
        if not manual_dir.is_dir():
            continue
        source = SOURCE_ALIASES.get(image_dir.parent.name, image_dir.parent.name)
        for scan in sorted(image_dir.iterdir()):
            if not scan.is_file() or scan.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask = manual_dir / f"{scan.stem}_manual.png"
            if not mask.is_file():
                raise ValueError(f"Missing manual mask for {scan}")
            with Image.open(scan) as image, Image.open(mask) as label:
                if image.size != label.size:
                    raise ValueError(f"Image and mask sizes differ: {scan}, {mask}")
                image.verify()
                label.verify()
            pairs.append({
                "scan": scan.relative_to(data_dir).as_posix(),
                "label": mask.relative_to(data_dir).as_posix(),
                "source": source,
            })
    if not pairs:
        raise ValueError(f"No BBBC019 image/manual pairs found below {data_dir}")
    names = [(pair["source"], Path(pair["scan"]).stem.casefold()) for pair in pairs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate BBBC019 image identifiers: {duplicates[:5]}")
    return pairs


def make_source_stratified_split(pairs, train_fraction=0.70, val_fraction=0.15, seed=42):
    """Split each of the eight source datasets so all remain represented."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than 1")
    by_source = defaultdict(list)
    for pair in pairs:
        if "source" not in pair:
            raise ValueError("Every BBBC019 pair must contain a source")
        by_source[pair["source"]].append(pair)
    split = {"train": [], "val": [], "test": []}
    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda pair: pair["scan"])
        random.Random(f"{seed}:{source}").shuffle(group)
        if len(group) < 3:
            raise ValueError(f"Source {source} needs at least three images")
        train_count = min(len(group) - 2, max(1, round(len(group) * train_fraction)))
        val_count = min(len(group) - train_count - 1, max(1, round(len(group) * val_fraction)))
        split["train"].extend(group[:train_count])
        split["val"].extend(group[train_count:train_count + val_count])
        split["test"].extend(group[train_count + val_count:])
    for name in split:
        split[name].sort(key=lambda pair: (pair["source"], pair["scan"]))
    split.update({
        "dataset": "BBBC019v2",
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "source_counts": {
            name: dict(sorted(Counter(pair["source"] for pair in split[name]).items()))
            for name in ("train", "val", "test")
        },
    })
    return split


def make_source_holdout_split(
    pairs, val_sources=DEFAULT_VAL_SOURCES, test_sources=DEFAULT_TEST_SOURCES, seed=42
):
    """Create source-disjoint splits for an honest cross-domain evaluation."""
    val_sources, test_sources = set(val_sources), set(test_sources)
    if not val_sources or not test_sources or val_sources & test_sources:
        raise ValueError("Validation and test sources must be nonempty and disjoint")
    available = {pair.get("source") for pair in pairs}
    requested = val_sources | test_sources
    if None in available or not requested <= available:
        raise ValueError(f"Requested sources {sorted(requested)} not found in {sorted(available)}")
    train_sources = available - requested
    if not train_sources:
        raise ValueError("At least one source must remain for training")
    split = {"train": [], "val": [], "test": []}
    for pair in pairs:
        source = pair["source"]
        name = "val" if source in val_sources else "test" if source in test_sources else "train"
        split[name].append(pair)
    for name in split:
        split[name].sort(key=lambda pair: (pair["source"], pair["scan"]))
    split.update({
        "dataset": "BBBC019v2",
        "split_strategy": "source_disjoint",
        "seed": seed,
        "train_sources": sorted(train_sources),
        "val_sources": sorted(val_sources),
        "test_sources": sorted(test_sources),
        "source_counts": {
            name: dict(sorted(Counter(pair["source"] for pair in split[name]).items()))
            for name in ("train", "val", "test")
        },
    })
    return split


def main():
    parser = argparse.ArgumentParser(description="Download and split BBBC019v2")
    parser.add_argument("--data-dir", type=Path, default=Path("data/BBBC019"))
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("splits/bbbc019.json"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-strategy", choices=("holdout", "stratified"), default="holdout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = (discover_bbbc019_pairs(args.data_dir) if args.skip_download
             else download_bbbc019(args.data_dir, args.archive_dir))
    split = (make_source_holdout_split(pairs, seed=args.seed)
             if args.split_strategy == "holdout"
             else make_source_stratified_split(
                 pairs, args.train_fraction, args.val_fraction, args.seed
             ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print({name: len(split[name]) for name in ("train", "val", "test")})
    print(f"Saved split to {args.output}")


if __name__ == "__main__":
    main()
