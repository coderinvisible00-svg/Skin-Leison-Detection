# Skin Lesion Analysis App (educational)

Kaggle-trained EfficientNet-B0 (HAM10000) + FastAPI backend + React dashboard.
Not a medical device. Not for diagnosis.

```
skin-cancer-app/
  backend/
    main.py
    requirements.txt
    model/               <- put best_model.pt, classes.json, metrics.json here
  frontend-files/
    src/App.jsx          <- copy into your Vite app
    src/App.css
```

## 1. Copy the Kaggle outputs

Download from your Kaggle notebook's Output tab and place in `backend/model/`:
`best_model.pt`, `classes.json`, `metrics.json`

## 2. Run the backend

```
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check http://localhost:8000/health and http://localhost:8000/docs

## 3. Create and run the frontend

From the project root (`skin-cancer-app/`):

```
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install recharts
```

Then:
1. Replace `frontend/src/App.jsx` and `frontend/src/App.css` with the files in `frontend-files/src/`.
2. Empty the contents of `frontend/src/index.css` (the default styles clash with the dashboard).

```
npm run dev
```

Open http://localhost:5173, upload an image, click Analyze.

## Troubleshooting

- "Cannot reach the backend": backend not running, or port differs. Set `VITE_API_URL` in `frontend/.env`.
- CORS error: the frontend must run on http://localhost:5173 (edit `allow_origins` in `main.py` otherwise).
- `best_model.pt` load error: it must be the state_dict saved by the training script (EfficientNet-B0, 7 classes).
