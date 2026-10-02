from pathlib import Path

import cv2

from vehicle_detector import VehicleDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "ultralytics"
    / "assets"
    / "bus.jpg"
)


def main():
    frame = cv2.imread(str(IMAGE_PATH))

    if frame is None:
        print(f"ERROR: Could not read image: {IMAGE_PATH}")
        return

    detector = VehicleDetector()
    detections = detector.detect(frame)

    print(f"\nTotal vehicles detected: {len(detections)}")
    print("-" * 50)

    for i, detection in enumerate(detections, start=1):
        print(
            f"{i}. "
            f"{detection['class_name']} | "
            f"Confidence: {detection['confidence']:.2f} | "
            f"BBox: "
            f"({detection['x1']}, {detection['y1']}, "
            f"{detection['x2']}, {detection['y2']})"
        )


if __name__ == "__main__":
    main()