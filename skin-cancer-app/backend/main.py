import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).parent / "model"

# ---- Load config, metrics and weights (files come from your Kaggle output) ----
cfg = json.loads((MODEL_DIR / "classes.json").read_text())
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())

class_names = cfg["class_names"]
full_names = cfg["full_names"]

# Group labels for display (HAM10000 class codes)
RISK = {
    "mel": "Malignant",
    "bcc": "Malignant",
    "akiec": "Pre-cancerous",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

# ---- API ----
app = FastAPI(title="Skin Lesion Classifier (educational)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image.")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()

    ranked = sorted(
        (
            {
                "code": c,
                "label": full_names.get(c, c),
                "probability": p,
                "group": RISK.get(c, "Benign"),
            }
            for c, p in zip(class_names, probs)
        ),
        key=lambda d: d["probability"],
        reverse=True,
    )
    top = ranked[0]
    return {
        "prediction": top["code"],
        "label": top["label"],
        "group": top["group"],
        "confidence": top["probability"],
        "probabilities": ranked,
    }
