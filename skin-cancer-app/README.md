# Skin Lesion Analysis Dashboard

An end-to-end deep learning project: a model trained on the HAM10000 dermatoscopic dataset, served through a FastAPI backend, with a React dashboard for uploading an image and viewing the prediction and training results.

> **Disclaimer:** Educational project only. It is not a medical device and must not be used for diagnosis. Always consult a qualified doctor.

![Dashboard screenshot](docs/dashboard.png)

## Features

- Upload an image (drag and drop) and get a predicted class with confidence
- Probability breakdown across all 7 lesion classes
- Model performance dashboard: accuracy, macro F1, training curves, confusion matrix, per-class precision/recall
- Light and dark theme

## Tech stack

| Part | Tools |
|---|---|
| Training | PyTorch, torchvision (EfficientNet-B0), scikit-learn, Kaggle GPU |
| Backend | FastAPI, Uvicorn |
| Frontend | React (Vite), Recharts |

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) ("Human Against Machine"): 10,015 dermatoscopic images across 7 classes:

| Code | Class |
|---|---|
| akiec | Actinic keratosis |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevus |
| vasc | Vascular lesion |

The dataset is heavily imbalanced (most images are nevi), so training uses class-weighted loss and model selection by validation macro-F1.

## Method

1. Split by `lesion_id` (about 67/17/17 train/val/test) so the same lesion never appears in two splits.
2. Pretrained EfficientNet-B0 with the classifier head replaced for 7 classes.
3. Augmentation: flips, rotation, colour jitter. AdamW optimizer, cosine learning-rate schedule, mixed precision.
4. Best checkpoint chosen by validation macro-F1; final numbers reported on the held-out test set.

## Results

| Metric | Value |
|---|---|
| Test accuracy | _fill in_ |
| Test macro F1 | _fill in_ |
| Melanoma recall | _fill in_ |

Add the confusion matrix and training curves from `training/` here.

## Project structure

```
skin-cancer-app/
  training/        Kaggle training script / notebook
  backend/         FastAPI app (main.py, requirements.txt, model/)
  frontend/        React (Vite) dashboard
  docs/            Screenshots
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status |
| GET | `/metrics` | Training and test results |
| POST | `/predict` | Multipart image upload; returns prediction and class probabilities |

## Limitations

- Trained on one dataset of dermatoscopic images; phone photos or other skin tones and body sites may perform much worse.
- Class imbalance: rare classes have fewer test samples, so their metrics are noisy.
- Confidence scores are not calibrated probabilities of being correct.

## Future work

- Grad-CAM explanations
- Probability calibration
- External validation (e.g. ISIC 2019/2020)
- Deployed demo

## Acknowledgements

Tschandl et al., HAM10000 dataset. Built with PyTorch, FastAPI, React and Recharts.
