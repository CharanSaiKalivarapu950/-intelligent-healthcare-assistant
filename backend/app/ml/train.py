import argparse
import os
import random
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from backend.app.ml.data import make_loaders
from backend.app.ml.model import create_model

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), all_labels, all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/raw/chest_xray")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/pneumonia_resnet18.pth")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    loaders, datasets_by_split = make_loaders(args.data, args.batch_size)
    model = create_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": datasets_by_split["train"].classes,
                "image_size": 224,
                "architecture": "resnet18",
            }, args.output)
            print(f"Saved best checkpoint: {args.output}")

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, labels, preds = run_epoch(model, loaders["test"], criterion, device)
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=checkpoint["class_names"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
