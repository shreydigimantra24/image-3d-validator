"""
Background Removal Service — uses RMBG-2.0 from BriaAI via HuggingFace.
"""

import os
import uuid
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Cache the model globally so it's loaded only once
_model = None


def _get_model():
    """Lazy-load and cache the RMBG-2.0 model."""
    global _model
    if _model is None:
        from transformers import AutoModelForImageSegmentation

        _model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
    return _model


# Preprocessing transform expected by RMBG-2.0
_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def remove_background(image_path: str, output_dir: str) -> dict:
    """
    Remove the background from a product image using RMBG-2.0.

    Args:
        image_path: Path to the input product image.
        output_dir: Directory where the output will be saved.

    Returns:
        Dictionary with output paths and metadata.
    """
    model = _get_model()
    device = next(model.parameters()).device

    # Load the image
    input_image = Image.open(image_path).convert("RGB")
    original_size = input_image.size  # (W, H)

    # Preprocess
    input_tensor = _transform(input_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        preds = model(input_tensor)[-1]

    # Post-process: sigmoid → resize to original → threshold
    pred_mask = torch.sigmoid(preds[0][0])
    pred_mask = pred_mask.cpu().numpy()

    # Resize mask back to original image size
    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize(original_size, Image.LANCZOS)

    # Apply mask to create RGBA output
    input_array = np.array(input_image)
    mask_array = np.array(mask_image)
    rgba = np.dstack([input_array, mask_array])
    output_image = Image.fromarray(rgba, "RGBA")

    # Save output
    file_id = str(uuid.uuid4())
    output_filename = f"{file_id}_rgba.png"
    output_path = os.path.join(output_dir, output_filename)
    output_image.save(output_path)

    return {
        "output_path": output_path,
        "output_url": f"/outputs/{output_filename}",
        "original_size": original_size,
        "output_size": output_image.size,
    }
