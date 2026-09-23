"""Aggregate segmentation metrics across an entire data loader."""

import torch
from torch import nn
import torch.nn.functional as F

from model import center_crop


def _metrics_from_counts(loss_sum, pixels, tp, fp, fn, tn, samples):
    return {
        "loss": loss_sum / pixels,
        "pixel_accuracy": (tp + tn) / pixels,
        "dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 1.0,
        "samples": samples,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = pixels = tp = fp = fn = tn = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        labels = center_crop(labels, *logits.shape[-2:])
        loss_sum += criterion(logits, labels).item()
        predictions = logits.argmax(dim=1)
        tp += ((predictions == 1) & (labels == 1)).sum().item()
        fp += ((predictions == 1) & (labels == 0)).sum().item()
        fn += ((predictions == 0) & (labels == 1)).sum().item()
        tn += ((predictions == 0) & (labels == 0)).sum().item()
        pixels += labels.numel()
    if not pixels:
        raise ValueError("Cannot evaluate an empty dataset")
    return _metrics_from_counts(loss_sum, pixels, tp, fp, fn, tn, len(loader.dataset))


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """Return aggregate, per-source, and per-image semantic metrics."""
    model.eval()
    totals = {"loss": 0.0, "pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "samples": 0}
    source_totals = {}
    per_image = []
    offset = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        labels = center_crop(labels, *logits.shape[-2:])
        predictions = logits.argmax(dim=1)
        for batch_index in range(len(images)):
            pair = loader.dataset.pairs[offset + batch_index]
            truth = labels[batch_index]
            prediction = predictions[batch_index]
            loss_sum = F.cross_entropy(
                logits[batch_index:batch_index + 1],
                truth[None], reduction="sum"
            ).item()
            counts = {
                "loss": loss_sum,
                "pixels": truth.numel(),
                "tp": ((prediction == 1) & (truth == 1)).sum().item(),
                "fp": ((prediction == 1) & (truth == 0)).sum().item(),
                "fn": ((prediction == 0) & (truth == 1)).sum().item(),
                "tn": ((prediction == 0) & (truth == 0)).sum().item(),
                "samples": 1,
            }
            for key in totals:
                totals[key] += counts[key]
            source = pair.get("source", "default")
            aggregate = source_totals.setdefault(
                source, {key: 0 for key in totals}
            )
            for key in aggregate:
                aggregate[key] += counts[key]
            metrics = _metrics_from_counts(
                counts["loss"], counts["pixels"], counts["tp"], counts["fp"],
                counts["fn"], counts["tn"], 1
            )
            per_image.append({
                "image": pair["scan"],
                "source": source,
                **metrics,
            })
        offset += len(images)
    if not totals["pixels"]:
        raise ValueError("Cannot evaluate an empty dataset")
    aggregate = _metrics_from_counts(
        totals["loss"], totals["pixels"], totals["tp"], totals["fp"],
        totals["fn"], totals["tn"], totals["samples"]
    )
    aggregate["mean_image_dice"] = sum(row["dice"] for row in per_image) / len(per_image)
    aggregate["per_source"] = {
        source: _metrics_from_counts(
            values["loss"], values["pixels"], values["tp"], values["fp"],
            values["fn"], values["tn"], values["samples"]
        )
        for source, values in sorted(source_totals.items())
    }
    aggregate["per_image"] = per_image
    return aggregate
