"""FastAPI backend for the PPE Safety Detection website."""

from pathlib import Path
from uuid import uuid4
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
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
    VIDEO_FRAME_STRIDE,
)
from .detector import PPEDetector, DetectorError
from .compliance import summarize_compliance
from .video_processor import process_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe-api")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PPE Safety Detection API",
    version="1.1.0",
    description="PPE detection for uploaded images and videos.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PPEDetector(MODEL_PATH)

# Job status is stored in memory. A service restart clears active job statuses.
video_jobs = {}

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


@app.get("/", include_in_schema=False)
def home():
    path = FRONTEND_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html was not found.",
        )
    return FileResponse(str(path))


@app.get("/style.css", include_in_schema=False)
def frontend_css():
    path = FRONTEND_DIR / "style.css"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="style.css not found.")
    return FileResponse(path, media_type="text/css")


@app.get("/script.js", include_in_schema=False)
def frontend_js():
    path = FRONTEND_DIR / "script.js"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="script.js not found.")
    return FileResponse(path, media_type="application/javascript")


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
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
                    )

                target.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        return destination

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()


@app.get("/api/health")
def health():
    state = detector.status()

    return {
        "status": "ok" if state["model_loaded"] else "model_missing",
        "model_loaded": state["model_loaded"],
        "model_path": state["model_path"],
        "device": state["device"],
        "load_error": state["load_error"],
    }


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    source = await save_upload(file, ALLOWED_IMAGE_EXTENSIONS)
    output = OUTPUT_DIR / f"{source.stem}_annotated.jpg"

    try:
        detections, width, height = detector.predict(source)
        detector.save_annotated_image(source, output)

        return {
            "filename": file.filename,
            "image_width": width,
            "image_height": height,
            "detection_count": len(detections),
            "detections": detections,
            "compliance": summarize_compliance(detections),
            "annotated_image_url": f"/outputs/{output.name}",
        }

    except DetectorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        source.unlink(missing_ok=True)


def run_video_job(job_id: str, source: Path, output: Path, filename: str):
    """Process the video after returning the job ID to the browser."""

    video_jobs[job_id]["status"] = "processing"
    video_jobs[job_id]["started_at"] = time.time()

    try:
        logger.info("Video job %s started: %s", job_id, filename)

        report = process_video(
            source,
            output,
            detector,
            frame_stride=VIDEO_FRAME_STRIDE,
        )

        video_jobs[job_id].update({
            "status": "completed",
            "result": {
                **report,
                "filename": filename,
                "output_url": f"/outputs/{output.name}",
            },
            "finished_at": time.time(),
        })

        logger.info("Video job %s completed", job_id)

    except Exception as exc:
        logger.exception("Video job %s failed", job_id)

        video_jobs[job_id].update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": time.time(),
        })

    finally:
        source.unlink(missing_ok=True)


@app.post("/api/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    source = await save_upload(file, ALLOWED_VIDEO_EXTENSIONS)

    job_id = uuid4().hex
    output = OUTPUT_DIR / f"{job_id}_annotated.mp4"

    video_jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        run_video_job,
        job_id,
        source,
        output,
        file.filename or "uploaded_video",
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video accepted. Check the job status for progress.",
    }


@app.get("/api/analyze/video/{job_id}")
def get_video_job(job_id: str):
    job = video_jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Video job not found. The server may have restarted; submit the video again.",
        )

    response = {
        "job_id": job_id,
        "status": job["status"],
    }

    if job["status"] == "completed":
        response["result"] = job["result"]

    if job["status"] == "failed":
        response["error"] = job.get("error", "Video processing failed.")

    return response


@app.get("/api")
def api_home():
    return {
        "message": "PPE Safety Detection API is running",
        "website": "/",
        "documentation": "/docs",
        "health_check": "/api/health",
    }