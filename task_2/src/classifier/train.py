import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision import datasets, transforms


def prepare_data(data_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_full   = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    num_classes = len(train_full.classes)
    total = len(train_full)
    logging.info(f"There are {total} images and {num_classes} classes.")

    # Split indices — same split for both, but different transforms applied
    indices = list(range(total))
    split = int(0.8 * total)

    train_dataset = Subset(train_full, indices[:split])
    val_dataset   = Subset(val_full,   indices[split:])

    # Calculate class weights to handle dataset imbalance.
    # We compute this strictly on the training subset to prevent data leakage.
    # Formula used: W = N_total / (N_classes * N_class)
    train_targets = [train_full.targets[i] for i in indices[:split]]
    class_counts = np.bincount(train_targets)
    class_weights = total * 0.8 / (num_classes * np.where(class_counts == 0, 1, class_counts))
    class_weights_tensor = torch.FloatTensor(class_weights)

    logging.info(f"Calculated Class Weights: {class_weights_tensor}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, class_weights_tensor, train_full.classes


def build_model(num_classes=10, freeze_features=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False  # freeze backbone

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on the Animals-10 dataset")
    parser.add_argument('--data_dir', type=str, default='data/Animals-10', help='Path to the dataset directory')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader, class_weights, class_names = prepare_data(args.data_dir, args.batch_size)

    class_weights = class_weights.to(device)

    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logging.info("-" * 15)

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.model_dir, exist_ok=True)
            save_path = os.path.join(args.model_dir, "best_animal_classifier.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names    
            }, save_path)
            logging.info(f"Model saved to {save_path}")

    logging.info(f"\nBest Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()