import base64
import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def make_gradcam_overlay(
    model,
    image: Image.Image,
    device,
    target_class: int | None = None,
):
    target_layer = model.layer4[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.update({"value": grad})
        )

    hook = target_layer.register_forward_hook(forward_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        logits = model(tensor)

        if target_class is None:
            target_class = int(
                logits.argmax(dim=1).item()
            )

        score = logits[0, target_class]

        score.backward()

        acts = activations["value"][0].detach()
        grads = gradients["value"][0].detach()

        weights = grads.mean(dim=(1, 2))

        cam = torch.relu(
            (weights[:, None, None] * acts).sum(dim=0)
        )

        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_image = Image.fromarray(
            np.uint8(cam * 255)
        ).resize(
            image.size,
            Image.Resampling.BILINEAR,
        )

        heatmap = np.asarray(cam_image).astype(
            np.float32
        ) / 255.0

        original = np.asarray(
            image.convert("RGB")
        ).astype(np.float32)

        color_heatmap = np.zeros_like(original)

        color_heatmap[..., 0] = heatmap * 255
        color_heatmap[..., 1] = heatmap * 80

        overlay = np.uint8(
            np.clip(
                0.65 * original + 0.35 * color_heatmap,
                0,
                255,
            )
        )

        output = io.BytesIO()

        Image.fromarray(overlay).save(
            output,
            format="PNG",
        )

        return base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)