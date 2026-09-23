"""Create the loss plot and portfolio-ready experiment report."""

import json
import csv
from pathlib import Path


def save_loss_plot(history, path, test_loss=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    ax.plot(epochs, [row["loss"] for row in history], marker="o", label="Validation")
    if test_loss is not None:
        ax.axhline(test_loss, color="tab:green", linestyle="--",
                   label="Held out test (evaluated once)")
    ax.set(xlabel="Epoch", ylabel="Cross entropy loss", title="Segmentation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(summary, metrics, preview_names, output_dir):
    output_dir = Path(output_dir)
    if not summary or not metrics:
        raise ValueError("Training summary and test metrics are required for a report")
    previews = "\n".join(
        f"![Scan, ground truth, prediction for {Path(name).stem}]({name})"
        for name in preview_names
    ) or "No previews were requested."
    architecture_note = (
        "No architecture deviation was used in this run."
        if summary["base_channels"] == 64
        else f"The first block used {summary['base_channels']} channels instead of the assignment's 64."
    )
    best_validation_dice = summary.get("best_validation_dice")
    best_validation_dice_row = (
        f"| Best validation Dice | {best_validation_dice:.4f} |\n"
        if best_validation_dice is not None else ""
    )
    source_rows = ""
    if metrics.get("per_source"):
        rows = ["| Source | Images | Dice | IoU | Precision | Recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for source, values in metrics["per_source"].items():
            rows.append(
                f"| {source} | {values['samples']} | {values['dice']:.4f} | "
                f"{values['iou']:.4f} | {values['precision']:.4f} | {values['recall']:.4f} |"
            )
        source_rows = "\n## Performance by BBBC019 source\n\n" + "\n".join(rows) + "\n"
    dataset_note = ""
    if summary.get("dataset") == "BBBC019v2":
        used_pairs = sum(
            sum(summary.get("source_counts", {}).get(name, {}).values())
            for name in ("train", "val", "test")
        )
        pair_note = (
            f" This run used {used_pairs} verified image-mask pairs."
            if used_pairs else ""
        )
        dataset_note = (
            "\n## Dataset attribution\n\n"
            "This experiment uses [BBBC019v2](https://bbbc.broadinstitute.org/BBBC019) "
            "from the Broad Bioimage Benchmark Collection. The official page catalogs 171 "
            "manually segmented DIC microscopy images across eight source datasets."
            f"{pair_note} The dataset is licensed "
            "under CC BY 3.0. The dataset files are not redistributed in this repository.\n"
        )
    split_note = ""
    if summary.get("split_strategy") == "source_disjoint":
        split_note = (
            "\nThe split is source-disjoint: training uses "
            f"{', '.join(summary['train_sources'])}; validation uses "
            f"{', '.join(summary['val_sources'])}; and the held-out test uses "
            f"{', '.join(summary['test_sources'])}. This measures transfer to acquisition "
            "sources not seen during training.\n"
        )
    report = f"""# Cell segmentation results

This report was generated from a completed run of the assignment-style U-Net. The split was fixed before training. Validation selected the checkpoint; the held out test set was evaluated once.
{split_note}

## Configuration

| Setting | Value |
| --- | ---: |
| Dataset | {summary.get('dataset', 'custom paired dataset')} |
| Input image size | {summary['image_size']} x {summary['image_size']} |
| Output mask size | {summary['output_height']} x {summary['output_width']} |
| Train / validation / test images | {summary['train_samples']} / {summary['val_samples']} / {summary['test_samples']} |
| Epochs | {summary['epochs']} |
| Best validation epoch | {summary['best_epoch']} |
{best_validation_dice_row}| Batch size | {summary['batch_size']} |
| Learning rate | {summary['learning_rate']} |
| Device | {summary['device']} |
| Training time | {summary['training_seconds']:.1f} seconds |

The network uses unpadded 3 x 3 convolutions, transpose-convolution upsampling, and center-cropped skip connections. {architecture_note} Images are grayscale and normalized to [0, 1]. Training augmentation uses horizontal and vertical flips, 90-degree rotations, zooming, and gamma correction.

## Loss and test metrics

The chart shows training and validation loss at every epoch. The horizontal test line is the single held out test evaluation, not a per-epoch test measurement.

![Training, validation, and held out test loss](loss_curves.png)

| Held out test metric | Value |
| --- | ---: |
| Cross entropy loss | {metrics['loss']:.4f} |
| Pixel accuracy | {metrics['pixel_accuracy']:.4f} |
| Dice | {metrics['dice']:.4f} |
| Intersection over union | {metrics['iou']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| Mean per-image Dice | {metrics.get('mean_image_dice', metrics['dice']):.4f} |
| Test images | {metrics['samples']} |

{source_rows}
{dataset_note}

## Test segmentations

Each preview shows the centered scan crop, ground-truth mask, and prediction from left to right.

{previews}

The complete epoch history and raw metrics are saved in `history.json` and `metrics.json`. The exact split is also saved in `split.json` when that file is present in the exported results.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_per_image_csv(rows, path):
    fields = ["image", "source", "loss", "pixel_accuracy", "dice", "iou",
              "precision", "recall", "samples"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
