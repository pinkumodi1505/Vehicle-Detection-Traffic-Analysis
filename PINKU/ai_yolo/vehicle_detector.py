from ultralytics import YOLO


class VehicleDetector:
    """
    YOLO11n-based vehicle detector.

    Detects:
        2 -> car
        3 -> motorcycle
        5 -> bus
        7 -> truck
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.50,
        image_size=640,
        iou=0.50,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou

    def detect(self, frame):
        """
        Detect vehicles in one OpenCV BGR frame.

        Returns:
            list of dictionaries containing:
            class_name, confidence, x1, y1, x2, y2
        """

        results = self.model(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
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