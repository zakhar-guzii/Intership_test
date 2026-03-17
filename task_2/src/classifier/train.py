import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(
    data_dir: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, torch.Tensor, list[str]]:
    """
    Load dataset from `data_dir`, apply augmentations on the train split,
    compute class weights to address imbalance, and return data loaders.

    Class weight formula: W_c = N_train / (C * N_c)
    where N_train = train subset size, C = number of classes, N_c = samples in class c.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load twice — same indices, different transforms
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_full   = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    num_classes = len(train_full.classes)
    total       = len(train_full)
    logging.info(f"Dataset: {total} images across {num_classes} classes.")

    # Shuffle indices with fixed seed for reproducibility
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int((1.0 - val_split) * total)
    train_indices = indices[:split]
    val_indices   = indices[split:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset   = Subset(val_full,   val_indices)

    logging.info(f"Split: {len(train_indices)} train / {len(val_indices)} val")

    # Class weights computed only on the train subset (no data leakage)
    train_targets  = [train_full.targets[i] for i in train_indices]
    class_counts   = np.bincount(train_targets, minlength=num_classes)
    class_weights  = len(train_indices) / (
        num_classes * np.where(class_counts == 0, 1, class_counts)
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    logging.info(f"Class weights: {class_weights_tensor.tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_weights_tensor, train_full.classes



def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """
    Load pretrained ResNet-18 and replace the classification head.

    If `freeze_backbone` is True, only the final FC layer is trained (fast
    feature extraction). Set to False to fine-tune the entire network.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        logging.info("Backbone frozen — training classification head only.")
    else:
        logging.info("Full fine-tuning — all layers trainable.")

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes),
    )

    return model



def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-18 classifier on the Animals-10 dataset."
    )
    parser.add_argument("--data_dir",       type=str,   default="data/Animals-10",
                        help="Path to dataset root directory.")
    parser.add_argument("--model_dir",      type=str,   default="models/",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--epochs",         type=int,   default=20,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size",     type=int,   default=32,
                        help="Batch size for training and validation.")
    parser.add_argument("--learning_rate",  type=float, default=1e-3,
                        help="Initial learning rate for the optimizer.")
    parser.add_argument("--val_split",      type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2).")
    parser.add_argument("--num_workers",    type=int,   default=2,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--patience",       type=int,   default=5,
                        help="Early-stopping patience (epochs without improvement).")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze ResNet backbone; train only the classification head.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")

    # Data
    train_loader, val_loader, class_weights, class_names = prepare_data(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        val_split   = args.val_split,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    # Model
    model = build_model(
        num_classes     = len(class_names),
        freeze_backbone = args.freeze_backbone,
    ).to(device)

    # Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop with early stopping
    best_val_acc      = 0.0
    epochs_no_improve = 0
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, "best_animal_classifier.pth")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")
        logging.info("-" * 40)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step()

        logging.info(f"Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        logging.info(f"Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "class_names":      class_names,
                    "args":             vars(args),
                },
                save_path,
            )
            logging.info(f"Checkpoint saved → {save_path}  (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            logging.info(
                f"No improvement for {epochs_no_improve}/{args.patience} epochs."
            )
            if epochs_no_improve >= args.patience:
                logging.info("Early stopping triggered.")
                break

    logging.info(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()