import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from backend.app.ml.data import make_loaders
from backend.app.ml.model import create_model
from backend.app.ml.train import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="datasets/raw/chest_xray")
parser.add_argument("--checkpoint", default="models/pneumonia_resnet18.pth")
args = parser.parse_args()

device = get_device()
loaders, datasets_by_split = make_loaders(args.data, batch_size=32)
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = create_model(pretrained=False, num_classes=len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()
labels, preds, probs = [], [], []
with torch.no_grad():
    for images, y in loaders["test"]:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1).cpu()
        preds.extend(p.argmax(dim=1).tolist())
        probs.extend(p[:, 1].tolist())
        labels.extend(y.tolist())
print(classification_report(labels, preds, target_names=checkpoint["class_names"], zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(labels, preds))
try:
    print("ROC-AUC:", roc_auc_score(labels, probs))
except ValueError as exc:
    print("ROC-AUC unavailable:", exc)
