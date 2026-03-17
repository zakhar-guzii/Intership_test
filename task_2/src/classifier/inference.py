import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError


def build_head(num_ftrs: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes),
    )


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    """
    Load a checkpoint saved by train.py.

    Expected checkpoint keys:
        model_state_dict, class_names, val_acc, epoch, args
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict  = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names", None)
        val_acc     = checkpoint.get("val_acc", None)
        epoch       = checkpoint.get("epoch", None)
        if val_acc is not None:
            logging.info(f"Checkpoint — epoch: {epoch}, val_acc: {val_acc:.4f}")
    else:
        logging.warning(
            "Legacy checkpoint format detected (raw state_dict). "
            "Class names will fall back to the built-in default list."
        )
        state_dict  = checkpoint
        class_names = None

    if class_names is None:
        class_names = [
            "butterfly", "cat", "chicken", "cow", "dog",
            "elephant", "horse", "sheep", "spider", "squirrel",
        ]
        logging.warning(f"Using default class list: {class_names}")

    num_classes = len(class_names)
    model = models.resnet18(weights=None)
    model.fc = build_head(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    return model.to(device), class_names


VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(
    image_path: str,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """
    Run inference on a single image.

    Returns a list of (class_name, confidence) tuples, sorted descending,
    of length min(top_k, num_classes).
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        logging.error(f"Cannot open image '{image_path}': {exc}")
        sys.exit(1)

    tensor = VAL_TRANSFORMS(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    top_k  = min(top_k, len(class_names))
    values, indices = torch.topk(probs, top_k)

    return [
        (class_names[idx.item()], round(val.item() * 100, 2))
        for val, idx in zip(values, indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained Animals-10 ResNet-18 classifier."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to best_animal_classifier.pth checkpoint.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image file to classify.",
    )
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Number of top predictions to display (default: 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model, class_names = load_model(args.model_path, device)
    results = predict(args.image_path, model, class_names, device, top_k=args.top_k)

    print("\n" + "=" * 40)
    print(f"  Image : {args.image_path}")
    print("=" * 40)
    if args.top_k == 1:
        name, conf = results[0]
        print(f"  Predicted : {name.upper()}  ({conf:.1f}%)")
    else:
        print("  Top predictions:")
        for rank, (name, conf) in enumerate(results, start=1):
            print(f"  {rank}. {name:<12} {conf:.1f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()