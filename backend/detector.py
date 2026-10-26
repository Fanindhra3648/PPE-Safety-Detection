
"""YOLO detector with NVIDIA GPU acceleration."""

from pathlib import Path
from threading import Lock

import torch

from backend.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASS_NAMES,
)


class DetectorError(RuntimeError):
    pass


class PPEDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None
        self._lock = Lock()
        self._load_error = None

        # Automatically use NVIDIA GPU when available.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.model_path.exists():
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.model_path))
                self._model.to(self.device)

            except Exception as exc:
                self._load_error = str(exc)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "model_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "device": self.device,
            "load_error": self._load_error,
        }

    def predict(self, image_source) -> tuple[list[dict], int, int]:
        """Predict from an image path or an OpenCV image array."""

        if self._model is None:
            raise DetectorError(
                f"Model weights unavailable: {self.model_path}. "
                f"Load error: {self._load_error}"
            )

        try:
            # Avoid converting OpenCV frames into temporary JPG files.
            source = (
                str(image_source)
                if isinstance(image_source, (str, Path))
                else image_source
            )

            with self._lock:
                results = self._model.predict(
                    source=source,
                    device=self.device,
                    imgsz=640,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=self.device.startswith("cuda"),
                    verbose=False,
                )

            if not results:
                raise DetectorError("No inference result returned.")

            result = results[0]
            height, width = result.orig_shape
            detections = []
            names = getattr(result, "names", {})

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = [
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ]

                    class_name = names.get(
                        class_id,
                        CLASS_NAMES.get(class_id, f"class_{class_id}")
                    )

                    detections.append({
                        "class_id": class_id,
                        "class_name": str(class_name),
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
                f"Inference failed: {exc}"
            ) from exc

    def save_annotated_image(
        self, image_path: Path, output_path: Path
    ) -> None:
        """Run detection and save an annotated image."""

        if self._model is None:
            raise DetectorError("Model weights are not available.")

        try:
            with self._lock:
                results = self._model.predict(
                    source=str(image_path),
                    device=self.device,
                    imgsz=640,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=self.device.startswith("cuda"),
                    save=False,
                    verbose=False,
                )

            if not results:
                raise DetectorError("No annotated image was returned.")

            import cv2

            output_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = results[0].plot()

            if not cv2.imwrite(str(output_path), annotated):
                raise DetectorError(
                    "Could not save the annotated image."
                )

        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"Could not create annotated image: {exc}"
            ) from exc
