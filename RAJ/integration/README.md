# RAJ — Core Integration

## Role

Core Integration Developer

## Responsibility

Responsible for integrating the project modules and managing the main application workflow.

## Responsibilities

- Integrate the YOLO detection module with the application.
- Implement video-based vehicle detection.
- Implement live webcam detection.
- Implement ByteTrack-based vehicle tracking.
- Implement cumulative line-crossing vehicle counting.
- Accept the counting-line position selected by the user.
- Display FPS and performance information.
- Connect the Image Processing module with the AI / YOLO module.
- Connect the detection and counting system with the UI module.
- Manage the overall application workflow.
- Ensure that the different modules work together correctly.

## Module Input

The Core Integration module receives:

- Preprocessed OpenCV BGR frames.
- Standardized vehicle detections from the YOLO detection module.
- User-selected horizontal counting-line position (`line_y`).

Example:

```python
line_y = 450

Module Output
The Core Integration module provides:
- Processed video frames.
- Stable vehicle tracking IDs.
- Cumulative vehicle counts.
- Per-class vehicle counts.
- FPS.
- Processing status.
Example:
{    "total_count": 6,    "class_counts": {        "car": 5,        "motorcycle": 0,        "bus": 0,        "truck": 1    },    "fps": 20.7,    "status": "completed"}


Counting-Line Interface
The counting line is selected by the user through the UI.
The UI provides the selected vertical position:
line_y


The Core Integration module receives this value:
processor = VideoProcessor(    detector=detector,    line_y=line_y,    frame_rate=30)


The selected line is then used by the vehicle counter for cumulative line-crossing detection.
The Core Integration module does not decide the counting-line position itself.

Integration Flow

Image Processing
↓
AI / YOLO Detection
↓
ByteTrack Tracking
↓
Line-Crossing Counting
↓
Processed Video + Counts + FPS + Status
↓
UI / Dashboard

The Core Integration module acts as the central connection between the different project modules.

Technology
- Python
- OpenCV
- YOLO / Ultralytics
- ByteTrack
- NumPy
