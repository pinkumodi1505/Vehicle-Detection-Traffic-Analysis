import time

import cv2

from src.tracking.tracker import VehicleTracker
from src.counting.counter import VehicleCounter


class VideoProcessor:
    """
    Processes a video frame by frame.

    Pipeline:
        Input Frame
            ↓
        Detector
            ↓
        ByteTrack
            ↓
        Line-Crossing Counter
            ↓
        Output Frame + Counting Data
    """

    def __init__(
        self,
        detector,
        line_y=300,
        frame_rate=30
    ):
        """
        Args:
            detector:
                Detector object that provides standardized detections.

            line_y:
                Y-coordinate of the counting line.

            frame_rate:
                Video frame rate used by ByteTrack.
        """

        self.detector = detector

        self.tracker = VehicleTracker(
            frame_rate=frame_rate
        )

        self.counter = VehicleCounter(
            line_y=line_y
        )

    def process_frame(self, frame):
        """
        Process a single OpenCV BGR frame.

        Args:
            frame:
                OpenCV BGR NumPy ndarray.

        Returns:
            processed_frame:
                Frame with tracking/counting information.

            counting_data:
                Standard project counting output.
        """

        # -----------------------------------
        # 1. Detection
        # -----------------------------------

        detections = self.detector.detect(frame)

        # -----------------------------------
        # 2. Tracking
        # -----------------------------------

        tracked_detections = self.tracker.update(
            detections
        )

        # -----------------------------------
        # 3. Counting
        # -----------------------------------

        counting_data = self.counter.update(
            tracked_detections
        )

        # -----------------------------------
        # 4. Draw results on frame
        # -----------------------------------

        processed_frame = self._draw_results(
            frame,
            tracked_detections
        )

        return processed_frame, counting_data

    def process_video(
        self,
        input_path,
        output_path
    ):
        """
        Process an entire video.

        Args:
            input_path:
                Path to the input video.

            output_path:
                Path where the processed video will be saved.

        Returns:
            Final counting information.
        """

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {input_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Default project output format: MP4
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0
        start_time = time.time()

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                # Process one frame
                processed_frame, counting_data = (
                    self.process_frame(frame)
                )

                # Calculate current FPS
                frame_count += 1

                elapsed_time = (
                    time.time() - start_time
                )

                if elapsed_time > 0:
                    current_fps = (
                        frame_count / elapsed_time
                    )
                else:
                    current_fps = 0.0

                # Update FPS and status
                counting_data["fps"] = round(
                    current_fps,
                    1
                )

                counting_data["status"] = "running"

                # Draw information
                processed_frame = self._draw_information(
                    processed_frame,
                    counting_data
                )

                # Write processed frame
                writer.write(
                    processed_frame
                )

        finally:
            cap.release()
            writer.release()

        # Return final result
        final_counts = self.counter.get_counts(
            fps=(
                frame_count /
                (time.time() - start_time)
                if time.time() > start_time
                else 0.0
            ),
            status="completed"
        )

        return final_counts

    def _draw_results(
        self,
        frame,
        tracked_detections
    ):
        """
        Draw bounding boxes and track IDs.
        """

        output_frame = frame.copy()

        for detection in tracked_detections:

            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])

            track_id = detection["track_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display class, confidence and ID
            label = (
                f"{class_name} "
                f"{confidence:.2f} "
                f"ID:{track_id}"
            )

            cv2.putText(
                output_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return output_frame

    def _draw_information(
        self,
        frame,
        counting_data
    ):
        """
        Draw counting information on the video.
        """

        output_frame = frame.copy()

        # Draw counting line
        cv2.line(
            output_frame,
            (0, self.counter.line_y),
            (
                output_frame.shape[1],
                self.counter.line_y
            ),
            (255, 0, 0),
            2
        )

        # Total count
        cv2.putText(
            output_frame,
            f"Total: {counting_data['total_count']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Class counts
        y_position = 60

        for class_name, count in (
            counting_data["class_counts"].items()
        ):

            cv2.putText(
                output_frame,
                f"{class_name}: {count}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y_position += 25

        # FPS
        cv2.putText(
            output_frame,
            f"FPS: {counting_data['fps']:.1f}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return output_frame