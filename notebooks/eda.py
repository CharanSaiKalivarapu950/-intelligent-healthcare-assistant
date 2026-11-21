from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

DATA = Path("datasets/raw/chest_xray")
for split in ["train", "val", "test"]:
    print(f"\\n{split.upper()}")
    counts = Counter()
    sizes = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for path in (DATA / split / cls).glob("*"):
            if path.is_file():
                counts[cls] += 1
                try:
                    with Image.open(path) as img:
                        sizes.append(img.size)
                except Exception:
                    pass
    print(counts)
    if sizes:
        print("Dimensions sample:", sizes[:10])
        print("Unique dimensions (first 20):", Counter(sizes).most_common(20))

# Show a small sample grid.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for row, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    paths = list((DATA / "train" / cls).glob("*"))[:5]
    for col, path in enumerate(paths):
        axes[row, col].imshow(Image.open(path).convert("RGB"), cmap="gray")
        axes[row, col].set_title(cls)
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()
