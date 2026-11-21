import os

import torch
from PIL import Image
from torchvision import transforms

from backend.app.core.config import settings
from backend.app.ml.model import create_model
from backend.app.ml.gradcam import make_gradcam_overlay


class Classifier:

    def __init__(self, checkpoint_path=None):

        self.checkpoint_path = (
            checkpoint_path or settings.model_path
        )

        self.device = self._device()

        self.model = None

        self.class_names = [
            "NORMAL",
            "PNEUMONIA",
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.load()

    @staticmethod
    def _device():

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self):

        if not os.path.isfile(self.checkpoint_path):
            return

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.class_names = checkpoint.get(
            "class_names",
            self.class_names,
        )

        self.model = create_model(
            pretrained=False,
            num_classes=len(self.class_names),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(self, image: Image.Image):

        if not self.is_loaded:
            raise FileNotFoundError(
                f"Trained checkpoint not found at "
                f"'{self.checkpoint_path}'. Train the model first."
            )

        tensor = self.transform(
            image.convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().tolist()

        index = int(
            torch.argmax(logits, dim=1).item()
        )

        return {
            "class_index": index,
            "predicted_class": self.class_names[index],
            "probabilities": {
                name: float(probability)
                for name, probability in zip(
                    self.class_names,
                    probabilities,
                )
            },
            "model": "ResNet18 transfer learning",
        }

    def generate_gradcam(
        self,
        image: Image.Image,
        target_class=None,
    ):

        if not self.is_loaded:
            raise FileNotFoundError(
                "Model checkpoint not loaded."
            )

        return make_gradcam_overlay(
            model=self.model,
            image=image,
            device=self.device,
            target_class=target_class,
        )


classifier = Classifier()