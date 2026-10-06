# PINKU — AI / YOLO

## Role

**AI / YOLO Developer**

## Responsibility

Responsible for developing the Artificial Intelligence and YOLO-based vehicle detection module of the Vehicle Detection & Traffic Analysis project.

The AI / YOLO module acts as the core computer-vision detection component of the project. It identifies vehicles from input images and video frames and provides structured detection information to the other project modules.

### 1. Pre-trained YOLO Model Setup

- Set up and configure the pre-trained YOLO model.
- Load and initialize the YOLO model using the Ultralytics framework.
- Configure the model for vehicle detection.
- Ensure that the detection model can process images and video frames.

### 2. Vehicle Detection

- Implement vehicle detection from input images and video frames.
- Process YOLO model outputs and extract valid detections.
- Filter the detected objects to identify the vehicle classes required by the project.

### 3. Vehicle Classification

Implement vehicle classification for:

- Car
- Motorcycle
- Bus
- Truck

The module identifies the class of each detected vehicle and provides this information to the other project modules.

### 4. Bounding Boxes

- Generate bounding boxes around detected vehicles.
- Extract and provide bounding-box coordinates for each detection.
- Make bounding-box information available for visualization by the UI module.

Example:

```python
{
    "class": "car",
    "bbox": [120, 80, 350, 300]
}

Input Image / Video Frame
            ↓
       YOLO Model
            ↓
     Object Detection
            ↓
    Vehicle Class Filter
            ↓
 ┌──────────┬────────────┬──────────┐
 ↓          ↓            ↓          ↓
Car    Motorcycle       Bus       Truck
 └──────────┴────────────┴──────────┘
            ↓
     Bounding Box
            +
      Confidence Score
            ↓
   Structured Detection
        Results