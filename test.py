"""Evaluate a saved checkpoint and export a few segmentation previews."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from metrics import evaluate_detailed
from model import UNet, center_crop
from preprocessing import Cell_data, load_split
from reporting import save_loss_plot, write_json, write_per_image_csv, write_report


def save_previews(model, dataset, device, output_dir, limit):
    names = []
    model.eval()
    with torch.no_grad():
        for index in range(min(limit, len(dataset))):
            image, label = dataset[index]
            tensor = torch.from_numpy(image[None]).to(device)
            prediction = model(tensor).argmax(dim=1)[0].cpu().numpy()
            height, width = prediction.shape
            image = center_crop(image[0], height, width)
            label = center_crop(label, height, width)
            panels = [
                Image.fromarray((image * 255).astype(np.uint8)),
                Image.fromarray((label * 255).astype(np.uint8)),
                Image.fromarray((prediction * 255).astype(np.uint8)),
            ]
            preview = Image.new("L", (width * 3, height))
            for column, panel in enumerate(panels):
                preview.paste(panel, (column * width, 0))
            stem = Path(dataset.pairs[index]["scan"]).stem
            source = dataset.pairs[index].get("source")
            prefix = f"{source}_" if source else ""
            name = f"{prefix}{stem}_preview.png"
            preview.save(output_dir / name)
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained cell segmentation checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("split.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/default/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-previews", type=int, default=0,
                        help="Maximum previews to save; 0 saves every test image")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_previews < 0:
        parser.error("batch size must be positive and max previews cannot be negative")

    split = load_split(args.split, args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if split != checkpoint["split"]:
        raise ValueError("The supplied split differs from the checkpoint's training split")
    dataset = Cell_data(args.data_dir, split["test"], checkpoint["image_size"], augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model = UNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate_detailed(model, loader, device)
    result["checkpoint_epoch"] = checkpoint["epoch"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", result)
    write_per_image_csv(result["per_image"], args.output_dir / "per_image_metrics.csv")
    limit = len(dataset) if args.max_previews == 0 else args.max_previews
    previews = save_previews(model, dataset, device, args.output_dir, limit)
    run_dir = args.checkpoint.parent
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    save_loss_plot(history, args.output_dir / "loss_curves.png", result["loss"])
    shutil.copyfile(args.split, args.output_dir / "split.json")
    shutil.copyfile(run_dir / "history.json", args.output_dir / "history.json")
    write_json(args.output_dir / "training_summary.json", summary)
    write_report(summary, result, previews, args.output_dir)
    print(json.dumps(result, indent=2))
    print(f"Saved metrics and previews to {args.output_dir}")


if __name__ == "__main__":
    main()
