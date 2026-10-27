import cv2
import numpy as np


def validate_frame(frame):
    """Check whether the input frame is valid."""
    if frame is None:
        return False

    if not isinstance(frame, np.ndarray):
        return False

    if frame.size == 0:
        return False

    return True


def resize_frame(frame, width=640):
    """Resize frame while maintaining aspect ratio."""
    if not validate_frame(frame):
        raise ValueError("Invalid frame")

    height, old_width = frame.shape[:2]

    if old_width == width:
        return frame

    ratio = width / old_width
    new_height = int(height * ratio)

    return cv2.resize(frame, (width, new_height))


def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
    """Adjust brightness and contrast."""
    if not validate_frame(frame):
        raise ValueError("Invalid frame")

    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def reduce_noise(frame):
    """Apply lightweight noise reduction."""
    if not validate_frame(frame):
        raise ValueError("Invalid frame")

    return cv2.GaussianBlur(frame, (3, 3), 0)


def preprocess_frame(
    frame,
    width=640,
    brightness=1.0,
    contrast=0
):
    """Prepare a frame for the YOLO detection module."""

    if not validate_frame(frame):
        raise ValueError("Invalid frame")

    frame = resize_frame(frame, width)

    if brightness != 1.0 or contrast != 0:
        frame = adjust_brightness_contrast(
            frame,
            brightness,
            contrast
        )

    frame = reduce_noise(frame)

    return frame


def extract_frames(video_path, frame_interval=1):
    """Extract frames from a video at a fixed interval."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    return frames