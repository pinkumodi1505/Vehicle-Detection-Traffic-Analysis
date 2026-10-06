from types import SimpleNamespace

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from src.config import VEHICLE_CLASSES


class VehicleTracker:
    """
    Tracks detected vehicles using ByteTrack.

    Input:
        List of standardized detection dictionaries:
        {
            "class_name": "car",
            "confidence": 0.87,
            "x1": 120,
            "y1": 80,
            "x2": 420,
            "y2": 300
        }

    Output:
        List of standardized detections with a stable track_id.
    """

    def __init__(self, frame_rate=30):
        self.frame_rate = frame_rate

        # ByteTrack parameters from the project standard
        tracker_args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
        )

        self.tracker = BYTETracker(
            tracker_args,
            frame_rate=frame_rate
        )

    def update(self, detections):
        """
        Update vehicle tracks for one video frame.

        Args:
            detections: List of standardized detection dictionaries.

        Returns:
            List of standardized detections with track_id.
        """

        if not detections:
            return []

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            # Convert XYXY to XYWH
            width = x2 - x1
            height = y2 - y1

            center_x = x1 + width / 2
            center_y = y1 + height / 2

            boxes.append([
                center_x,
                center_y,
                width,
                height
            ])

            confidences.append(
                detection["confidence"]
            )

            # Convert class name to standard class ID
            class_id = next(
                (
                    class_id
                    for class_id, class_name in VEHICLE_CLASSES.items()
                    if class_name == detection["class_name"]
                ),
                -1
            )

            class_ids.append(class_id)

        # IMPORTANT:
        # Ultralytics 8.3.110 ByteTrack requires NumPy arrays here.
        results = SimpleNamespace(
            xywh=np.asarray(
                boxes,
                dtype=np.float32
            ),
            conf=np.asarray(
                confidences,
                dtype=np.float32
            ),
            cls=np.asarray(
                class_ids,
                dtype=np.float32
            )
        )

        # Run ByteTrack
        tracked_objects = self.tracker.update(results)

        tracked_detections = []

        for tracked in tracked_objects:
            (
                x1,
                y1,
                x2,
                y2,
                track_id,
                confidence,
                class_id,
                detection_index
            ) = tracked

            detection_index = int(detection_index)

            # Get the original standardized detection
            original_detection = detections[detection_index]

            tracked_detections.append({
                "class_name": original_detection["class_name"],
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "track_id": int(track_id)
            })

        return tracked_detections