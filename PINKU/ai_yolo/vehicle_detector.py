from typing import Any

from ultralytics import YOLO


class VehicleDetector:
    """
    YOLO11n-based vehicle detector.

    Detects only the project-approved vehicle classes:
        2 -> car
        3 -> motorcycle
        5 -> bus
        7 -> truck

    The detector returns clean detection dictionaries for the
    tracking module. Tracking and counting are intentionally
    not handled here.
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.50,
        image_size: int = 640,
        iou: float = 0.50,
    ) -> None:
        """
        Initialize the YOLO vehicle detector.

        Args:
            model_path: Path to the YOLO11n model.
            confidence: Minimum confidence threshold.
            image_size: YOLO inference image size.
            iou: IoU threshold for NMS.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """
        Detect vehicles in one OpenCV BGR frame.

        Args:
            frame: OpenCV BGR image as a NumPy array.

        Returns:
            A list of dictionaries containing:

                class_name
                confidence
                x1
                y1
                x2
                y2

            No track_id is returned because tracking belongs
            to the RAJ module.
        """
        if frame is None:
            return []

        results = self.model(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0],
                )

                detections.append(
                    {
                        "class_name": self.VEHICLE_CLASSES[class_id],
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        return detections