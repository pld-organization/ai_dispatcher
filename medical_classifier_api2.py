import os
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

# ── CONFIG ───────────────────────────────────────────

LABEL_MAP = {
    0: "bone_cancer",
    1: "breast",
    2: "lung_cell",
    3: "colon_cell",
    4: "bone_fracture",
    5: "brain",
    6: "lung",
    7: "liver",
    8: "skin",
    9: "blood",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

_model: Optional[nn.Module] = None


# ── MODEL ───────────────────────────────────────────

def load_model(model_path: str) -> nn.Module:
    global _model
    if _model is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, 10)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        _model = model.to(DEVICE)

    return _model


# ── FILE → PIL ──────────────────────────────────────

def load_any_to_pil(path: str) -> Image.Image:
    ext = Path(path).suffix.lower()

    if ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

    elif ext in (".nii", ".gz"):
        import nibabel as nib
        vol = nib.load(path).get_fdata()
        arr = vol[:, :, vol.shape[2] // 2].astype(np.float32)

    elif ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path).astype(np.float32)

    else:
        return Image.open(path).convert("RGB")

    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


# ── CLASSIFY FROM FILE ──────────────────────────────

@torch.no_grad()
def classify_image(path: str, model_path: str) -> Dict:
    model = load_model(model_path)

    img = load_any_to_pil(path)
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    class_id = int(np.argmax(probs))

    return {
        "label": LABEL_MAP[class_id],
        "class_id": class_id,
        "confidence": float(probs[class_id]),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }


# ── CLASSIFY FROM BYTES (FAST VERSION) ──────────────

@torch.no_grad()
def classify_image_bytes(file_bytes: bytes, filename: str, model_path: str) -> Dict:
    model = load_model(model_path)
    ext = Path(filename).suffix.lower()

    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    elif ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)

        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    elif ext in [".nii", ".gz"]:
        import nibabel as nib
        with tempfile.NamedTemporaryFile(suffix=filename) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            vol = nib.load(tmp.name).get_fdata()

        arr = vol[:, :, vol.shape[2] // 2].astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    elif ext in [".tif", ".tiff"]:
        import tifffile
        arr = tifffile.imread(io.BytesIO(file_bytes)).astype(np.float32)

        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    
    elif ext == ".zip":
        import zipfile
        SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".tif", ".tiff", ".nii", ".gz"}
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            candidates = [f for f in zf.namelist() if Path(f).suffix.lower() in SUPPORTED]
            if not candidates:
                raise ValueError("ZIP contains no supported medical image files")
            # classify the first valid image found inside
            img_bytes = zf.read(candidates[0])
            return classify_image_bytes(img_bytes, candidates[0], model_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)

    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    class_id = int(np.argmax(probs))

    return {
        "label": LABEL_MAP[class_id],
        "class_id": class_id,
        "confidence": float(probs[class_id]),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }