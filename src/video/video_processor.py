import time
from pathlib import Path

import cv2

from PRITAM.image_processing.preprocessor import preprocess_frame
from src.tracking.tracker import VehicleTracker
from src.counting.counter import VehicleCounter


class VideoProcessor:
    """
    Process video frames through detection, tracking, and line-crossing counting.

    Pipeline:
        Input frame
            -> optional Pritam preprocessing for detector input
            -> vehicle detector
            -> restore detections to original-frame coordinates
            -> ByteTrack
            -> line-crossing counter
            -> annotated output frame
    """

    def __init__(
        self,
        detector,
        line_y=300,
        frame_rate=30,
        use_preprocessing=False,
    ):
        """
        Args:
            detector:
                Detector object with detect(frame), returning detections
                as dictionaries containing x1, y1, x2, y2, class_name,
                confidence, and any additional fields required by the tracker.

            line_y:
                Y-coordinate of the counting line in original-frame pixels.

            frame_rate:
                Video frame rate used by the tracker.

            use_preprocessing:
                If True, apply Pritam's preprocess_frame() to the detector
                input. Disabled by default to preserve existing behaviour.
        """
        self.detector = detector
        self.use_preprocessing = use_preprocessing

        self.tracker = VehicleTracker(frame_rate=frame_rate)
        self.counter = VehicleCounter(line_y=line_y)

    def process_frame(self, frame):
        """
        Process one OpenCV BGR frame.

        Preprocessing is applied only to the detector's input. Tracking,
        counting, and annotation use coordinates in the original frame.

        Returns:
            (processed_frame, counting_data)
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        original_height, original_width = frame.shape[:2]
        detection_frame = frame

        # 1. Optional preprocessing from Pritam's module.
        if self.use_preprocessing:
            detection_frame = preprocess_frame(frame)

        detection_height, detection_width = detection_frame.shape[:2]

        # 2. Detect vehicles on the selected detector input.
        detections = self.detector.detect(detection_frame)

        # 3. Map bounding boxes back to original-frame coordinates if resized.
        # Keep the original detection dictionaries and other fields intact.
        if self.use_preprocessing:
            scale_x = original_width / detection_width
            scale_y = original_height / detection_height

            adjusted_detections = []
            for detection in detections:
                adjusted = detection.copy()
                for coordinate in ("x1", "x2"):
                    if coordinate in adjusted:
                        adjusted[coordinate] = float(adjusted[coordinate]) * scale_x
                for coordinate in ("y1", "y2"):
                    if coordinate in adjusted:
                        adjusted[coordinate] = float(adjusted[coordinate]) * scale_y
                adjusted_detections.append(adjusted)

            detections = adjusted_detections

        # 4. Track vehicles using original-frame coordinates.
        tracked_detections = self.tracker.update(detections)

        # 5. Count tracked vehicles crossing the configured line.
        counting_data = self.counter.update(tracked_detections)

        # 6. Draw annotations on the original-size frame.
        processed_frame = self._draw_results(frame, tracked_detections)

        return processed_frame, counting_data

    def process_video(self, input_path, output_path):
        """
        Process an entire video and save an annotated MP4.

        Args:
            input_path: Input video path.
            output_path: Destination path for the annotated video.

        Returns:
            Final counting information from VehicleCounter.
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError(f"Could not create output video: {output_path}")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                processed_frame, counting_data = self.process_frame(frame)
                frame_count += 1

                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0

                counting_data["fps"] = round(current_fps, 1)
                counting_data["status"] = "running"

                processed_frame = self._draw_information(
                    processed_frame,
                    counting_data,
                )
                writer.write(processed_frame)

        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start_time
        final_fps = frame_count / elapsed if elapsed > 0 else 0.0

        final_counts = self.counter.get_counts(
            fps=final_fps,
            status="completed",
        )
        return final_counts

    def _draw_results(self, frame, tracked_detections):
        """Draw bounding boxes, classes, confidence values, and track IDs."""
        output_frame = frame.copy()

        for detection in tracked_detections:
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])

            track_id = detection["track_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"{class_name} {confidence:.2f} ID:{track_id}"
            cv2.putText(
                output_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output_frame

    def _draw_information(self, frame, counting_data):
        """Draw the counting line, totals, class counts, and FPS."""
        output_frame = frame.copy()
        line_y = int(self.counter.line_y)

        cv2.line(
            output_frame,
            (0, line_y),
            (output_frame.shape[1] - 1, line_y),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            output_frame,
            f"Total: {counting_data['total_count']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y_position = 60
        for class_name, count in counting_data["class_counts"].items():
            cv2.putText(
                output_frame,
                f"{class_name}: {count}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_position += 25

        cv2.putText(
            output_frame,
            f"FPS: {counting_data['fps']:.1f}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output_frame
