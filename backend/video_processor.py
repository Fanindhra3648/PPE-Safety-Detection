"""Video processing with reduced memory usage."""

from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg

from .config import (
    VIDEO_FRAME_STRIDE,
    MAX_VIDEO_SECONDS,
    MAX_OUTPUT_WIDTH,
    MAX_DETECTION_RECORDS,
)
from .detector import DetectorError


VIOLATION_CLASSES = {"NoHelmet", "NoVest"}
SAFE_CLASSES = {"Helmet", "Vest"}


def resize_frame(frame):
    height, width = frame.shape[:2]

    if width <= MAX_OUTPUT_WIDTH:
        return frame

    new_height = int(height * MAX_OUTPUT_WIDTH / width)

    return cv2.resize(
        frame,
        (MAX_OUTPUT_WIDTH, new_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_annotations(frame, detections):
    has_violation = any(
        item.get("class_name", "") in VIOLATION_CLASSES
        for item in detections
    )

    has_ppe = any(
        item.get("class_name", "") in SAFE_CLASSES
        for item in detections
    )

    if has_violation:
        banner = "PPE VIOLATION DETECTED"
        color = (0, 0, 255)
    elif has_ppe:
        banner = "PPE DETECTED"
        color = (0, 160, 0)
    else:
        banner = "PPE NOT CONFIRMED"
        color = (0, 140, 255)

    height, width = frame.shape[:2]

    cv2.rectangle(
        frame, (0, 0), (width, min(38, height)), color, -1
    )

    cv2.putText(
        frame,
        banner,
        (10, min(26, height - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for item in detections:
        name = item.get("class_name", "Unknown")
        confidence = float(item.get("confidence", 0))

        x1 = max(0, min(width - 1, int(item["x1"])))
        y1 = max(0, min(height - 1, int(item["y1"])))
        x2 = max(0, min(width - 1, int(item["x2"])))
        y2 = max(0, min(height - 1, int(item["y2"])))

        box_color = (
            (0, 0, 255)
            if name in VIOLATION_CLASSES
            else (0, 160, 0)
        )

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), box_color, 2
        )

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(48, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def process_video(
    video_path,
    output_path,
    detector,
    frame_stride=VIDEO_FRAME_STRIDE,
):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if frame_stride < 1:
        raise ValueError("Frame stride must be at least 1.")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            "Cannot open video. Please upload a valid MP4 video."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 120:
        fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0

    if duration > MAX_VIDEO_SECONDS:
        capture.release()
        raise ValueError(
            f"Video is {duration:.0f} seconds long. "
            f"Maximum allowed duration is {MAX_VIDEO_SECONDS} seconds."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_name(
        output_path.stem + "_temp.avi"
    )

    frame_index = 0
    frames_processed = 0
    frames_with_violations = 0
    violation_count = 0
    no_helmet_count = 0
    no_vest_count = 0
    detection_records = []
    last_detections = []
    any_safe_detection = False

    writer = None

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame = resize_frame(frame)

            height, width = frame.shape[:2]

            if writer is None:
                # MJPG is used only as a temporary file.
                writer = cv2.VideoWriter(
                    str(temp_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise ValueError(
                        "Could not create temporary video output."
                    )

            if frame_index % frame_stride == 0:
                detections, _, _ = detector.predict(frame)

                last_detections = detections
                frames_processed += 1

                if any(
                    d.get("class_name", "") in VIOLATION_CLASSES
                    for d in detections
                ):
                    frames_with_violations += 1

                for item in detections:
                    name = item.get("class_name", "Unknown")

                    if name in SAFE_CLASSES:
                        any_safe_detection = True

                    if name in VIOLATION_CLASSES:
                        violation_count += 1

                    if name == "NoHelmet":
                        no_helmet_count += 1
                    elif name == "NoVest":
                        no_vest_count += 1

                    if len(detection_records) < MAX_DETECTION_RECORDS:
                        record = dict(item)
                        record["frame_number"] = frame_index + 1
                        record["timestamp_seconds"] = round(
                            frame_index / fps, 2
                        )
                        detection_records.append(record)

            annotated = draw_annotations(frame, last_detections)
            writer.write(annotated)
            frame_index += 1

    except DetectorError:
        raise
    finally:
        capture.release()

        if writer is not None:
            writer.release()

    if frame_index == 0 or writer is None:
        temp_path.unlink(missing_ok=True)
        raise ValueError("The uploaded video contains no readable frames.")

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-i", str(temp_path),
            "-an",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0 or not output_path.is_file():
            raise ValueError(
                "MP4 conversion failed: "
                + (result.stderr or "Unknown FFmpeg error")[-800:]
            )

    finally:
        temp_path.unlink(missing_ok=True)

    if violation_count:
        status = "VIOLATION"
    elif any_safe_detection:
        status = "SAFE"
    else:
        status = "UNKNOWN"

    return {
        "frames_processed": frames_processed,
        "total_frames_read": frame_index,
        "frames_in_video": frame_index,
        "frames_with_violations": frames_with_violations,
        "detection_count": sum(
            1 for _ in detection_records
        ),
        "violation_count": violation_count,
        "no_helmet_count": no_helmet_count,
        "no_vest_count": no_vest_count,
        "overall_status": status,
        "detections": detection_records,
        "detections_truncated": (
            len(detection_records) >= MAX_DETECTION_RECORDS
        ),
        "output_file": output_path.name,
    }