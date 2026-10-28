"""Central configuration for PPE Safety Detection."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "PPE_MODEL_PATH",
        str(
            PROJECT_ROOT
            / "runs"
            / "detect"
            / "ppe_train"
            / "weights"
            / "best.pt"
        ),
    )
)

UPLOAD_DIR = Path(
    os.getenv("PPE_UPLOAD_DIR", str(PROJECT_ROOT / "backend" / "uploads"))
)

OUTPUT_DIR = Path(
    os.getenv("PPE_OUTPUT_DIR", str(PROJECT_ROOT / "backend" / "outputs"))
)

MAX_UPLOAD_MB = int(os.getenv("PPE_MAX_UPLOAD_MB", "50"))
CONFIDENCE_THRESHOLD = float(
    os.getenv("PPE_CONFIDENCE_THRESHOLD", "0.25")
)
IOU_THRESHOLD = float(os.getenv("PPE_IOU_THRESHOLD", "0.45"))

INFERENCE_SIZE = int(os.getenv("PPE_INFERENCE_SIZE", "416"))
VIDEO_FRAME_STRIDE = max(
    1, int(os.getenv("PPE_VIDEO_FRAME_STRIDE", "4"))
)
MAX_VIDEO_SECONDS = int(os.getenv("PPE_MAX_VIDEO_SECONDS", "120"))
MAX_OUTPUT_WIDTH = int(os.getenv("PPE_MAX_OUTPUT_WIDTH", "640"))
MAX_DETECTION_RECORDS = int(
    os.getenv("PPE_MAX_DETECTION_RECORDS", "500")
)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

CLASS_NAMES = {
    0: "Helmet",
    1: "NoHelmet",
    2: "NoVest",
    3: "Vest",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)