# PPE Safety Detection

A structured starter project for a PPE computer-vision application. The current dataset classes are:

| Class ID | Class name | Meaning |
|---:|---|---|
| 0 | Helmet | Helmet detected |
| 1 | NoHelmet | No-helmet class detected |
| 2 | NoVest | No-safety-vest class detected |
| 3 | Vest | Safety vest detected |

> **Important:** this code does not include trained weights. Add `models/best.pt` after training. Until then, the API can start and report its health, but inference requests return HTTP 503. Detection results are only decision support; a human should verify safety decisions.

## 1. Project structure

```text
PPE_Safety_Detection/
├── backend/
│   ├── app.py
│   ├── compliance.py
│   ├── config.py
│   ├── detector.py
│   ├── schemas.py
│   ├── video_processor.py
│   ├── requirements.txt
│   ├── uploads/
│   └── outputs/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   └── README.md
├── tests/
│   └── test_compliance.py
├── .gitignore
├── render.yaml
└── README.md
```

Keep your existing `data/` and `notebooks/` folders. They are not included in this starter package so your dataset and notebook are not overwritten.

## 2. Setup on Windows PowerShell

Open PowerShell in the project root (the folder containing `README.md`).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed, use a Python version supported by the installed Ultralytics, PyTorch and OpenCV wheels.

## 3. Start the backend

From the project root:

```powershell
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open:
- API status: http://127.0.0.1:8000/api/health
- Interactive API documentation: http://127.0.0.1:8000/docs

Expected before model training: `"status": "model_missing"`. That is expected, not an API startup failure.

## 4. Open the frontend

For a reliable local setup, open a second terminal in the project root and run:

```powershell
python -m http.server 5500 --directory frontend
```

Then open http://127.0.0.1:5500.

The frontend expects the API at `http://127.0.0.1:8000`. If you deploy the frontend separately, set `window.PPE_API_BASE` to your backend URL before loading `script.js`, or edit `API_BASE` in `frontend/script.js`.

## 5. Run tests

```powershell
python -m pytest -q
```

These tests cover compliance summary logic; they do not test model accuracy.

## 6. Add trained weights later

After training and evaluating the YOLO model, copy the best weights to:

```text
models/best.pt
```

Restart the backend and check `/api/health`. It should report `model_loaded: true` if the weights load successfully.

## 7. Deployment notes

`render.yaml` describes a Python web service for the API. Before deploying:
1. Train and evaluate the model.
2. Decide how to provide `best.pt` to the service. Large model weights may be better stored in a model artifact or persistent external storage than committed to Git.
3. Restrict CORS to the exact frontend origin.
4. Confirm upload limits, memory use, video processing time, and the chosen Render plan.
5. Test the deployed `/api/health` and `/docs` endpoints.

Render's local filesystem may be ephemeral, so uploaded files and generated outputs should not be treated as permanent storage. The frontend is static files; deploy it as a static site or serve it from an appropriate web host.

## API endpoints

- `GET /` — API welcome message.
- `GET /api/health` — reports whether the model is loaded.
- `POST /api/analyze/image` — upload an image using form field `file`.
- `POST /api/analyze/video` — upload a video using form field `file`.

The video endpoint processes sampled frames to limit work. For long videos, production deployments should use a background job queue and a persistent result store.