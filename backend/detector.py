"""YOLO detector with automatic CPU/GPU selection."""

from pathlib import Path
from threading import Lock

import torch

from .config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    INFERENCE_SIZE,
    CLASS_NAMES,
)


class DetectorError(RuntimeError):
    """Raised when model loading or prediction fails."""


class PPEDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.lock = Lock()
        self.load_error = None

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda:0" if self.use_cuda else "cpu"

        if not self.model_path.is_file():
            self.load_error = (
                f"Model weights not found: {self.model_path}"
            )
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))

            if self.use_cuda:
                self.model.to("cuda:0")

        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def is_loaded(self):
        return self.model is not None

    def status(self):
        return {
            "model_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "device": self.device,
            "load_error": self.load_error,
        }

    def predict(self, image_source):
        if self.model is None:
            raise DetectorError(
                self.load_error or "Model is not loaded."
            )

        source = (
            str(image_source)
            if isinstance(image_source, (str, Path))
            else image_source
        )

        try:
            with self.lock:
                results = self.model.predict(
                    source=source,
                    device=self.device,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=self.use_cuda,
                    max_det=50,
                    verbose=False,
                    save=False,
                )

            if not results:
                raise DetectorError("Model returned no results.")

            result = results[0]
            height, width = result.orig_shape
            detections = []
            names = result.names or {}

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    x1, y1, x2, y2 = [
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ]

                    name = names.get(
                        class_id,
                        CLASS_NAMES.get(
                            class_id, f"class_{class_id}"
                        ),
                    )

                    detections.append({
                        "class_id": class_id,
                        "class_name": str(name),
                        "confidence": round(confidence, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    })

            return detections, int(width), int(height)

        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"Prediction failed: {type(exc).__name__}: {exc}"
            ) from exc

    def save_annotated_image(self, image_path, output_path):
        if self.model is None:
            raise DetectorError(
                self.load_error or "Model is not loaded."
            )

        import cv2

        try:
            with self.lock:
                results = self.model.predict(
                    source=str(image_path),
                    device=self.device,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=self.use_cuda,
                    max_det=50,
                    verbose=False,
                    save=False,
                )

            if not results:
                raise DetectorError(
                    "No image result returned by the model."
                )

            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            annotated = results[0].plot()

            if not cv2.imwrite(str(output_path), annotated):
                raise DetectorError(
                    "Could not save annotated image."
                )

        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"Image saving failed: {type(exc).__name__}: {exc}"
            ) from exc