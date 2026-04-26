# Medical Image Classification API

FastAPI service that accepts any medical image file, routes it through
the EfficientNet-B0 router, then passes it to the matching specialist model.

## Structure

```
medical_api/
├── main.py                     # Entrypoint
├── pending/                    # Temp storage — files live here while queued
│                               # Auto-deleted after processing
├── medical_router_v1.pth       # Router weights (copy here)
│
└── app/
    ├── api/
    │   └── routes.py           # FastAPI endpoints
    │
    ├── queue/
    │   └── manager.py          # Disk-backed FIFO queue, single worker thread
    │
    ├── ml/
    │   ├── router.py           # Router model loader + inference
    │   ├── dispatcher.py       # Router → specialist routing logic
    │   ├── specialists.py      # One handler per class (breast/lung wired, rest stubs)
    │   ├── model.py            # YOUR existing model loaders (unchanged)
    │   └── preprocessing.py    # YOUR existing preprocessors (unchanged)
    │
    ├── fileshandler/
    │   └── files.py            # YOUR existing file utils (unchanged)
    │
    └── core/
        ├── config.py           # YOUR existing settings (unchanged)
        └── images.py           # YOUR existing image utils (unchanged)
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/predict` | Auto-route via router model |
| POST | `/api/v1/predict?label=lung_cell` | Skip router, force specialist |
| POST | `/api/v1/predict/{label}` | Same, path-style override |
| GET  | `/api/v1/labels` | List valid label names |

## How a request flows

```
Upload file
    │
    ▼
Write to  pending/<job_id>.<ext>   ← disk, survives crashes
    │
    ▼
FIFO Queue  (one job at a time)
    │
    ▼
Router model  →  label + confidence
    │  (skipped if label_override set)
    ▼
Specialist handler  →  result dict
    │
    ▼
Delete pending/<job_id>.<ext>
    │
    ▼
Return JSON to caller
(caller sends original bytes + result to storage service)
```

## Adding a specialist

In `app/ml/specialists.py` replace the stub:

```python
def handle_bone_cancer(file_bytes: bytes, filename: str) -> dict:
    # load your model, run inference, return a dict
    ...
```

## Running

```bash
pip install fastapi uvicorn torch torchvision pydicom nibabel tifffile scikit-image
python main.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_WEIGHTS` | `medical_router_v1.pth` | Path to router weights |
