"""PPE Safety Detection FastAPI Application."""

from pathlib import Path
from uuid import uuid4
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    MAX_UPLOAD_MB,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MODEL_PATH,
)
from .detector import PPEDetector, DetectorError
from .compliance import summarize_compliance
from .video_processor import process_video

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe-api")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PPE Safety Detection API",
    description="Detect Helmet, NoHelmet, NoVest and Vest in images and videos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load detection model
detector = PPEDetector(MODEL_PATH)

# Display processed output files - IDI CHALA IMPORTANT
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)

# ---------------- FRONTEND WEBSITE ----------------
@app.get("/", include_in_schema=False)
def home():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail=f"Website file not found: {index_file}")
    return FileResponse(str(index_file))

@app.get("/style.css", include_in_schema=False)
def frontend_css():
    css_file = FRONTEND_DIR / "style.css"
    if not css_file.is_file():
        raise HTTPException(status_code=404, detail="style.css not found")
    return FileResponse(css_file, media_type="text/css")

@app.get("/script.js", include_in_schema=False)
def frontend_js():
    js_file = FRONTEND_DIR / "script.js"
    if not js_file.is_file():
        raise HTTPException(status_code=404, detail="script.js not found")
    return FileResponse(js_file, media_type="application/javascript")

# ---------------- SAVE UPLOADED FILE ----------------
async def save_upload(upload: UploadFile, allowed_extensions: set[str]) -> Path:
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in allowed_extensions:
        await upload.close()
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    destination = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    total_size = 0
    try:
        with destination.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
                target.write(chunk)
        if total_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

# ---------------- BACKEND STATUS ----------------
@app.get("/api/health")
def health():
    state = detector.status()
    return {
        "status": "ok" if state["model_loaded"] else "model_missing",
        "model_loaded": state["model_loaded"],
        "model_path": state["model_path"],
        "message": "Model loaded successfully." if state["model_loaded"] else "Backend is running, but trained weights are missing.",
    }

# ---------------- IMAGE ANALYSIS ----------------
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


# ---------------- VIDEO ANALYSIS ----------------
@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    source_path = await save_upload(file, ALLOWED_VIDEO_EXTENSIONS)
    output_path = OUTPUT_DIR / f"{source_path.stem}_annotated.mp4"

    try:
        report = process_video(
            source_path,
            output_path,
            detector,
            frame_stride=2,
        )

        return {
            **report,
            "filename": file.filename,
            "output_url": f"/outputs/{output_path.name}",
        }

    except DetectorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Video analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        ) from exc

    # IMPORTANT: Do not delete source_path here.
    # Original uploaded video must remain in backend/uploads/.


@app.get("/api")
def api_home():
    return {
        "message": "PPE Safety Detection API is running",
        "website": "/",
        "documentation": "/docs",
        "health_check": "/api/health",
    }