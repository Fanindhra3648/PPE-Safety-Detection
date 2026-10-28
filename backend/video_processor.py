
"""Memory-conscious video processing for PPE detection."""

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

# Conservative settings for low-memory hosting.
PROCESS_WIDTH = min(MAX_OUTPUT_WIDTH, 480)
PROCESS_STRIDE = max(VIDEO_FRAME_STRIDE, 8)
PROCESS_MAX_SECONDS = min(MAX_VIDEO_SECONDS, 30)


def resize_frame(frame):
    height, width = frame.shape[:2]

    if width <= PROCESS_WIDTH:
        return frame

    new_height = max(2, int(height * PROCESS_WIDTH / width))
    new_height -= new_height % 2

    return cv2.resize(
        frame,
        (PROCESS_WIDTH, new_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_annotations(frame, detections):
    height, width = frame.shape[:2]

    violation = any(
        d.get("class_name") in VIOLATION_CLASSES
        for d in detections
    )
    safe = any(
        d.get("class_name") in SAFE_CLASSES
        for d in detections
    )

    if violation:
        banner, color = "PPE VIOLATION DETECTED", (0, 0, 255)
    elif safe:
        banner, color = "PPE DETECTED", (0, 160, 0)
    else:
        banner, color = "PPE NOT CONFIRMED", (0, 140, 255)

    cv2.rectangle(frame, (0, 0), (width, min(38, height)), color, -1)
    cv2.putText(
        frame, banner, (10, min(26, height - 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    for item in detections:
        name = item.get("class_name", "Unknown")
        confidence = float(item.get("confidence", 0))
        x1 = max(0, min(width - 1, int(item["x1"])))
        y1 = max(0, min(height - 1, int(item["y1"])))
        x2 = max(0, min(width - 1, int(item["x2"])))
        y2 = max(0, min(height - 1, int(item["y2"])))

        box_color = (
            (0, 0, 255) if name in VIOLATION_CLASSES
            else (0, 160, 0)
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame, f"{name} {confidence:.2f}",
            (x1, max(48, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            box_color, 1, cv2.LINE_AA,
        )

    return frame


def process_video(
    video_path,
    output_path,
    detector,
    frame_stride=PROCESS_STRIDE,
):
    video_path = Path(video_path)
    output_path = Path(output_path)

    frame_stride = max(int(frame_stride), PROCESS_STRIDE)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Cannot open video. Please upload a valid video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0

    if duration > PROCESS_MAX_SECONDS:
        capture.release()
        raise ValueError(
            f"Video must be {PROCESS_MAX_SECONDS} seconds or shorter "
            "on this hosting plan. Please trim the video and retry."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pipe processed frames directly to FFmpeg; no temporary AVI.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "480x270",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "30",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    process = None
    frame_index = 0
    frames_processed = 0
    frames_with_violations = 0
    violation_count = 0
    no_helmet_count = 0
    no_vest_count = 0
    safe_detected = False
    detection_records = []
    last_detections = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_frame(frame)
            height, width = frame.shape[:2]

            # Keep a consistent even-sized frame for FFmpeg.
            out_width = min(width, 480)
            out_height = max(2, int(height * out_width / width))
            out_height -= out_height % 2

            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(
                    frame, (out_width, out_height),
                    interpolation=cv2.INTER_AREA,
                )

            if process is None:
                command[command.index("480x270")] = f"{out_width}x{out_height}"
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

            if frame_index % frame_stride == 0:
                detections, _, _ = detector.predict(frame)
                last_detections = detections
                frames_processed += 1

                if any(
                    d.get("class_name") in VIOLATION_CLASSES
                    for d in detections
                ):
                    frames_with_violations += 1

                for item in detections:
                    name = item.get("class_name", "Unknown")

                    if name in SAFE_CLASSES:
                        safe_detected = True
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
            process.stdin.write(annotated.tobytes())
            frame_index += 1

        if frame_index == 0:
            raise ValueError("The video contains no readable frames.")

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=300)

        if return_code != 0 or not output_path.is_file():
            raise ValueError("MP4 conversion failed: " + stderr[-500:])

    except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
        if process is not None and process.poll() is None:
            process.kill()
        raise ValueError(
            "Video encoding failed or timed out. Try a shorter video."
        ) from exc
    finally:
        capture.release()
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()

    if violation_count:
        status = "VIOLATION"
    elif safe_detected:
        status = "SAFE"
    else:
        status = "UNKNOWN"

    return {
        "frames_processed": frames_processed,
        "total_frames_read": frame_index,
        "frames_in_video": frame_index,
        "frames_with_violations": frames_with_violations,
        "detection_count": len(detection_records),
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
