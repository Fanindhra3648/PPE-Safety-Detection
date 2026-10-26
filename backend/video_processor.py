
"""PPE video processing with faster frame inference."""

from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg

from .detector import PPEDetector, DetectorError

VIOLATION_CLASSES = {"NoHelmet", "NoVest"}
SAFE_CLASSES = {"Helmet", "Vest"}


def detector_frame_plot(frame, detections):
    """Draw red violation boxes and a status banner."""

    has_violation = any(
        item.get("class_name", item.get("class", "")) in VIOLATION_CLASSES
        for item in detections
    )

    has_ppe = any(
        item.get("class_name", item.get("class", "")) in SAFE_CLASSES
        for item in detections
    )

    if has_violation:
        banner = "DETECTED: NO HELMET / NO VEST"
        color = (0, 0, 255)
    elif has_ppe:
        banner = "SAFE: PPE DETECTED"
        color = (0, 180, 0)
    else:
        banner = "CHECK: PPE NOT CONFIRMED"
        color = (0, 180, 255)

    cv2.rectangle(
        frame, (0, 0), (frame.shape[1], 48), color, -1
    )

    cv2.putText(
        frame,
        banner,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for item in detections:
        name = item.get("class_name", item.get("class", "Unknown"))
        confidence = float(item.get("confidence", 0))

        x1, y1, x2, y2 = (
            int(item[key]) for key in ("x1", "y1", "x2", "y2")
        )

        box_color = (
            (0, 0, 255) if name in VIOLATION_CLASSES else (0, 180, 0)
        )

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), box_color, 2
        )

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(65, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return frame, has_violation


def process_video(
    video_path: Path,
    output_path: Path,
    detector: PPEDetector,
    frame_stride: int = 2,
) -> dict:
    """
    Detect every Nth frame and write all frames to the output video.
    Keep the original upload; this function does not delete it.
    """

    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read the video's dimensions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_video = output_path.with_name(
        output_path.stem + "_temp.avi"
    )

    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create temporary output video.")

    frame_index = 0
    frames_processed = 0
    frames_with_violations = 0
    violation_count = 0
    no_helmet_count = 0
    no_vest_count = 0
    all_detections = []
    last_detections = []

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            # Run inference directly on the frame, without a temporary JPG.
            if frame_index % frame_stride == 0:
                detections, _, _ = detector.predict(frame)

                last_detections = detections
                frames_processed += 1

                current_has_violation = any(
                    item.get("class_name", item.get("class", ""))
                    in VIOLATION_CLASSES
                    for item in detections
                )

                if current_has_violation:
                    frames_with_violations += 1

                for item in detections:
                    name = item.get(
                        "class_name", item.get("class", "Unknown")
                    )

                    record = dict(item)
                    record["frame_number"] = frame_index + 1
                    record["timestamp_seconds"] = round(
                        frame_index / fps, 2
                    )
                    all_detections.append(record)

                    if name in VIOLATION_CLASSES:
                        violation_count += 1

                    if name == "NoHelmet":
                        no_helmet_count += 1
                    elif name == "NoVest":
                        no_vest_count += 1

            annotated, _ = detector_frame_plot(
                frame, last_detections
            )
            writer.write(annotated)
            frame_index += 1

    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("The uploaded video has no readable frames.")

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i", str(temp_video),
            "-an",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode != 0 or not output_path.is_file():
            raise ValueError(
                "Could not convert processed video to MP4. "
                + result.stderr[-1000:]
            )

    finally:
        temp_video.unlink(missing_ok=True)

    if violation_count > 0:
        overall_status = "VIOLATION"
    elif any(
        item.get("class_name", item.get("class", "")) in SAFE_CLASSES
        for item in all_detections
    ):
        overall_status = "SAFE"
    else:
        overall_status = "UNKNOWN"

    return {
        "frames_processed": frames_processed,
        "total_frames_read": frame_index,
        "frames_in_video": frame_index,
        "frames_with_violations": frames_with_violations,
        "detection_count": len(all_detections),
        "violation_count": violation_count,
        "no_helmet_count": no_helmet_count,
        "no_vest_count": no_vest_count,
        "overall_status": overall_status,
        "detections": all_detections,
        "output_file": output_path.name,
    }
