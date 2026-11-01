"""
Training Pipeline for Alzheimer's MRI Classification
Features:
  - Automatic Class Weighting for Imbalanced Medical Datasets
  - Cross-Entropy Loss with label smoothing
  - AdamW Optimizer with Cosine Annealing Learning Rate Scheduler
  - Model Checkpointing (saves best model based on validation loss)
  - Early Stopping to prevent overfitting
  - Training metrics logging
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from models.cnn_model import AlzheimerCNN
from utils.dataset import load_dataset_splits


def compute_class_weights(dataset, num_classes=4):
    """
    Computes balanced class weights inversely proportional to class frequencies.
    Crucial for handling imbalanced medical MRI datasets.
    """
    labels = [dataset.labels[i] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Inverse frequency weighting
    weights = total / (num_classes * (counts.astype(np.float32) + 1e-5))
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(data_dir="data/mri_dataset", epochs=15, batch_size=32, lr=5e-4,
                 checkpoint_dir="models/checkpoints", patience=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_dataset_splits(data_dir, batch_size=batch_size)

    # Compute class weights for medical imbalance
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    print(f"[INFO] Applying class balance weights: {class_weights.cpu().numpy().round(3)}")

    model = AlzheimerCNN(in_channels=3, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "best_alzheimer_model.pth")
    patience_counter = 0

    print("\n--- Starting Training Pipeline ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)
            print(f"  --> Saved new best model to {best_model_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} epochs.")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s.")
    print(f"Best checkpoint saved at: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI Classifier")
    parser.add_argument("--data_dir", type=str, default="data/mri_dataset", help="Path to MRI dataset root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    run_training(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
