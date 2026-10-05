class VehicleCounter:
    """
    Counts vehicles when their tracking point crosses a virtual line.

    The counter is cumulative:
    once a vehicle is counted, it will not be counted again.

    Input:
        Tracked detections from VehicleTracker:
        {
            "class_name": "car",
            "confidence": 0.87,
            "x1": 120,
            "y1": 80,
            "x2": 420,
            "y2": 300,
            "track_id": 15
        }

    Output:
        {
            "total_count": 12,
            "class_counts": {
                "car": 7,
                "motorcycle": 3,
                "bus": 1,
                "truck": 1
            },
            "fps": 18.6,
            "status": "running"
        }
    """

    def __init__(self, line_y=300):
        """
        Args:
            line_y: Y-coordinate of the virtual counting line.
        """

        self.line_y = line_y

        # Stores the previous center position of each tracked vehicle.
        self.previous_positions = {}

        # Stores track IDs that have already been counted.
        self.counted_ids = set()

        # Required vehicle classes from the project standard.
        self.class_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        self.total_count = 0

    def _get_center(self, detection):
        """
        Calculate the center point of a bounding box.
        """

        center_x = (
            detection["x1"] + detection["x2"]
        ) / 2

        center_y = (
            detection["y1"] + detection["y2"]
        ) / 2

        return center_x, center_y

    def update(self, tracked_detections):
        """
        Process tracked vehicles for one video frame.

        Args:
            tracked_detections:
                List of detections containing track_id.

        Returns:
            Dictionary containing cumulative counts.
        """

        for detection in tracked_detections:

            track_id = detection["track_id"]
            class_name = detection["class_name"]

            center_x, center_y = self._get_center(
                detection
            )

            # If this vehicle was seen in the previous frame,
            # check whether it crossed the counting line.
            if track_id in self.previous_positions:

                previous_x, previous_y = (
                    self.previous_positions[track_id]
                )

                # Vehicle crossed from above the line to below it.
                crossed_down = (
                    previous_y < self.line_y
                    and center_y >= self.line_y
                )

                # Vehicle crossed from below the line to above it.
                crossed_up = (
                    previous_y > self.line_y
                    and center_y <= self.line_y
                )

                crossed_line = crossed_down or crossed_up

                # Count each tracked vehicle only once.
                if (
                    crossed_line
                    and track_id not in self.counted_ids
                ):

                    self.counted_ids.add(track_id)

                    self.total_count += 1

                    if class_name in self.class_counts:
                        self.class_counts[class_name] += 1

            # Store current center for the next frame.
            self.previous_positions[track_id] = (
                center_x,
                center_y
            )

        return self.get_counts()

    def get_counts(self, fps=0.0, status="running"):
        """
        Return the current cumulative counting result.

        Args:
            fps: Current video FPS.
            status: Current processing status.

        Returns:
            Standard project counting output.
        """

        return {
            "total_count": self.total_count,
            "class_counts": self.class_counts.copy(),
            "fps": float(fps),
            "status": status
        }

    def reset(self):
        """
        Reset all counting information.
        """

        self.previous_positions.clear()
        self.counted_ids.clear()

        self.total_count = 0

        self.class_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }