"""Train a U-Net on a saved preprocessing split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import evaluate
from model import UNet, center_crop
from preprocessing import Cell_data, load_split
from reporting import save_loss_plot, write_json


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


def main():
    parser = argparse.ArgumentParser(description="Train cell segmentation U-Net")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=572)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8,
                        help="Stop after this many epochs without better validation Dice; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 0:
        parser.error("epochs, batch size, and learning rate must be positive; patience cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    split = load_split(args.split, args.data_dir)
    train_set = Cell_data(args.data_dir, split["train"], args.image_size, augment=True)
    val_set = Cell_data(args.data_dir, split["val"], args.image_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    device = choose_device(args.device)
    model = UNet(args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.json"
    history = []
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    output_shape = None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            output_shape = logits.shape[-2:]
            labels = center_crop(labels, *output_shape)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            samples += len(images)

        result = evaluate(model, val_loader, device)
        scheduler.step(result["loss"])
        record = {"epoch": epoch, "train_loss": loss_sum / samples,
                  "learning_rate": optimizer.param_groups[0]["lr"], **result}
        history.append(record)
        write_json(history_path, history)
        if result["dice"] > best_dice:
            best_dice = result["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "split": split,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch}/{args.epochs}: train loss {record['train_loss']:.4f}, "
            f"val loss {result['loss']:.4f}, Dice {result['dice']:.4f}, "
            f"IoU {result['iou']:.4f}, lr {record['learning_rate']:.2e}"
        )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break
    training_seconds = time.perf_counter() - start
    summary = {
        "dataset": split.get("dataset", "custom paired dataset"),
        "split_strategy": split.get("split_strategy", "image_level"),
        "train_sources": split.get("train_sources"),
        "val_sources": split.get("val_sources"),
        "test_sources": split.get("test_sources"),
        "image_size": args.image_size,
        "output_height": output_shape[0],
        "output_width": output_shape[1],
        "base_channels": args.base_channels,
        "epochs": len(history),
        "epochs_requested": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(split["test"]),
        "device": str(device),
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "source_counts": split.get("source_counts"),
    }
    write_json(args.output_dir / "training_summary.json", summary)
    save_loss_plot(history, args.output_dir / "loss_curves.png")
    print(f"Best checkpoint: {checkpoint_path} (Dice {best_dice:.4f})")
    print(f"Training time: {training_seconds:.1f} seconds; output mask: {output_shape}")


if __name__ == "__main__":
    main()
