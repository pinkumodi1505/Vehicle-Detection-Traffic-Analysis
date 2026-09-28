# PINKU - AI and YOLO Module
# PINKU - AI / YOLO Module

## Responsibility

This module handles the Artificial Intelligence and YOLO-based vehicle detection component of the Vehicle Detection & Traffic Analysis project.

## Main Tasks

- Configure and load the YOLO object detection model
- Detect vehicles from images and video frames
- Identify supported vehicle categories:
  - Car
  - Motorcycle
  - Bus
  - Truck
- Generate bounding boxes around detected vehicles
- Calculate and return confidence scores
- Provide a reusable detection function for other project modules

## Detection Flow

```text
Input Image / Video Frame
          ↓
      YOLO Model
          ↓
    Object Detection
          ↓
    Vehicle Filtering
          ↓
Car / Motorcycle / Bus / Truck
          ↓
Bounding Box + Confidence
          ↓
Structured Detection Results