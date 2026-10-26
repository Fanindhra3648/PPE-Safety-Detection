
"""Train a YOLOv8 Nano model for PPE safety detection."""

from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO


# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "ppe_dataset"
SOURCE_YAML = DATASET_DIR / "data.yaml"

# Training settings
MODEL_NAME = "yolov8n.pt"
EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 8

OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "ppe_train"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
BEST_MODEL = WEIGHTS_DIR / "best.pt"
FINAL_MODEL_DIR = PROJECT_ROOT / "models"
FINAL_MODEL = FINAL_MODEL_DIR / "best.pt"


def check_dataset():
    """Check dataset folders and labels before training."""

    if not SOURCE_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {SOURCE_YAML}"
        )

    with SOURCE_YAML.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_images = DATASET_DIR / config["train"]
    val_images = DATASET_DIR / config["val"]

    for folder in [train_images, val_images]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Required image folder not found: {folder}"
            )

        images = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        if not images:
            raise RuntimeError(
                f"No images found in: {folder}"
            )

        labels_folder = folder.parent / "labels"
        if not labels_folder.is_dir():
            raise FileNotFoundError(
                f"Labels folder not found: {labels_folder}"
            )

        print(f"{folder.name}: {len(images)} images")

    print("Classes:", config["names"])
    return config


def main():
    print("=" * 55)
    print("PPE SAFETY DETECTION - YOLO MODEL TRAINING")
    print("=" * 55)

    config = check_dataset()

    # Create an absolute-path configuration for reliable loading.
    training_config = dict(config)
    training_config["path"] = str(DATASET_DIR.resolve())

    temporary_yaml = PROJECT_ROOT / "data" / "ppe_training.yaml"
    temporary_yaml.parent.mkdir(parents=True, exist_ok=True)

    with temporary_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            training_config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\nLoading pretrained YOLO model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Training may take a while. Do not close the terminal.")

    model.train(
        data=str(temporary_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="ppe_train",
        exist_ok=True,
        patience=15,
        workers=0,
        seed=42,
        plots=True,
    )

    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt was not found at {BEST_MODEL}"
        )

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BEST_MODEL, FINAL_MODEL)

    print("\nTraining completed successfully.")
    print("Best trained weights:", BEST_MODEL)
    print("Backend model copy:", FINAL_MODEL)
    print("\nNext: evaluate the model and test a video.")


if __name__ == "__main__":
    main()
