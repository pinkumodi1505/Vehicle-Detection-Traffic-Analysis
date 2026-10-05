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

OUTPUT_PATH = PROJECT_ROOT / "PINKU" / "ai_yolo" / "output_bus_detection.jpg"


def main():
    # Read image
    frame = cv2.imread(str(IMAGE_PATH))

    if frame is None:
        print(f"ERROR: Could not read image: {IMAGE_PATH}")
        return

    # Create detector
    detector = VehicleDetector()

    # Detect vehicles
    detections = detector.detect(frame)

    # Draw detections
    for detection in detections:
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Label
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Save result
    cv2.imwrite(str(OUTPUT_PATH), frame)

    print(f"Vehicles detected: {len(detections)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()