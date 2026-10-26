"""Central application configuration."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "PPE_MODEL_PATH",
        str(PROJECT_ROOT / "runs" / "detect" / "ppe_train" / "weights" / "best.pt")
    )
)
UPLOAD_DIR = Path(os.getenv("PPE_UPLOAD_DIR", PROJECT_ROOT / "backend" / "uploads"))
OUTPUT_DIR = Path(os.getenv("PPE_OUTPUT_DIR", PROJECT_ROOT / "backend" / "outputs"))
MAX_UPLOAD_MB = int(os.getenv("PPE_MAX_UPLOAD_MB", "100"))
CONFIDENCE_THRESHOLD = float(os.getenv("PPE_CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("PPE_IOU_THRESHOLD", "0.45"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CLASS_NAMES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
