# PINKU — AI / YOLO

## Role

AI / YOLO Developer

## Responsibility

Responsible for developing the Artificial Intelligence and YOLO-based vehicle detection module of the Vehicle Detection & Traffic Analysis project.

## Responsibilities

- Set up and configure the pre-trained YOLO model.
- Implement vehicle detection from input images and video frames.
- Implement vehicle classification for:
  - Car
  - Motorcycle
  - Bus
  - Truck
- Generate bounding boxes around detected vehicles.
- Calculate and provide confidence scores.
- Create the core detection function that can be used by other project modules.
- Provide structured detection results to the integration module.
- Test and verify YOLO detection results.

## Module Output

- Vehicle class
- Bounding box coordinates
- Confidence score
- Detection results

## Integration

The AI / YOLO module provides detection results to the Core Integration module.

The results can also be used by the UI module for visualization and by the Testing module for performance analysis.

## Technology

- Python
- YOLO
- Ultralytics
- OpenCV
- NumPy
- PyTorch
