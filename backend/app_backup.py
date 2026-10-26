"""FastAPI application for PPE image analysis."""
from pathlib import Path
from uuid import uuid4
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import (
    UPLOAD_DIR, OUTPUT_DIR, MAX_UPLOAD_MB,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, MODEL_PATH,
)
from backend.detector import PPEDetector, DetectorError
from backend.compliance import summarize_compliance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe-api")

app = FastAPI(
    title="PPE Safety Detection API",
    description="Detect Helmet, NoHelmet, NoVest and Vest in uploaded images.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your deployed frontend domain before production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
detector = PPEDetector(MODEL_PATH)

async def save_upload(upload: UploadFile, allowed_extensions: set[str]) -> Path:
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}")
    destination = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    total = 0
    try:
        with destination.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return destination

@app.get("/")
def root():
    return {"message": "PPE Safety Detection API is running", "docs": "/docs"}

@app.get("/api/health")
def health():
    state = detector.status()
    return {
        "status": "ok" if state["model_loaded"] else "model_missing",
        "model_loaded": state["model_loaded"],
        "model_path": state["model_path"],
        "message": "Model loaded." if state["model_loaded"] else "Backend is running, but trained weights are not installed yet.",
    }

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    source_path = await save_upload(file, ALLOWED_IMAGE_EXTENSIONS)
    output_path = OUTPUT_DIR / f"{source_path.stem}_annotated.jpg"
    try:
        detections, width, height = detector.predict(source_path)
        detector.save_annotated_image(source_path, output_path)
        return {
            "filename": file.filename,
            "image_width": width,
            "image_height": height,
            "detection_count": len(detections),
            "detections": detections,
            "compliance": summarize_compliance(detections),
            "annotated_image_url": f"/outputs/{output_path.name}",
        }
    except DetectorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc
    finally:
        source_path.unlink(missing_ok=True)

@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    source_path = await save_upload(file, ALLOWED_VIDEO_EXTENSIONS)
    output_path = OUTPUT_DIR / f"{source_path.stem}_annotated.mp4"
    try:
        from backend.video_processor import process_video
        report = process_video(source_path, output_path, detector)
        return {**report, "output_url": f"/outputs/{output_path.name}"}
    except DetectorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
    finally:
        source_path.unlink(missing_ok=True)
