import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(pretrained: bool = True, num_classes: int = 2):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
