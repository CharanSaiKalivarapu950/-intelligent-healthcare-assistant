from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def build_transforms(train: bool):
    common = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose(common)

def make_loaders(data_dir: str, batch_size: int = 16, num_workers: int = 2):
    data_dir = Path(data_dir)
    required = ["train", "val", "test"]
    missing = [split for split in required if not (data_dir / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset folders: {missing}. Expected train/val/test.")
    datasets_by_split = {
        split: datasets.ImageFolder(data_dir / split, transform=build_transforms(split == "train"))
        for split in required
    }
    for split, dataset in datasets_by_split.items():
        if dataset.classes != CLASS_NAMES:
            raise ValueError(f"{split} classes are {dataset.classes}; expected {CLASS_NAMES}")
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, datasets_by_split
