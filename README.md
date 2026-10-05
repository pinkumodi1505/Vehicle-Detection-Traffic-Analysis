# Vehicle Detection & Traffic Analysis

## Project Overview

Vehicle Detection & Traffic Analysis is an AI-based computer vision project designed to detect and analyze vehicles from images, videos, and live camera input.

The system uses YOLO-based object detection to identify vehicles, generate bounding boxes, provide confidence scores, and support further traffic analysis.

## Project Objectives

- Detect vehicles automatically using YOLO.
- Classify detected vehicles.
- Generate bounding boxes around vehicles.
- Provide confidence scores for detections.
- Process images and video input.
- Support live webcam detection.
- Count detected vehicles.
- Display detection and traffic statistics.
- Analyze detection performance.

## Vehicle Classes

The system focuses on:

- Car
- Motorcycle
- Bus
- Truck

## Team Work Division

### PINKU — AI / YOLO

Responsible for:

- Pre-trained YOLO model setup
- Vehicle detection
- Vehicle classification
- Bounding boxes
- Confidence scores
- Core detection function

### PRITAM — Image Processing & I/O

Responsible for:

- Image preprocessing
- Resizing
- Brightness and contrast adjustment
- Noise reduction
- Frame extraction
- Input validation
- Saving processed outputs

### RAJ — Core Integration

Responsible for:

- YOLO integration
- Video detection
- Live webcam detection
- Vehicle counting
- FPS/performance display
- Connecting project modules

### SANSKRITI — UI & Visualization

Responsible for:

- Project interface
- Image/video/webcam selection
- Detection result display
- Vehicle statistics dashboard
- Bounding-box visualization
- User-friendly interface

### SANTANU — Testing & Performance

Responsible for:

- Testing different images and videos
- Testing different traffic and lighting conditions
- Checking missed and incorrect detections
- Measuring FPS and performance
- Accuracy/result analysis
- Bug identification and coordination

### DEBANSHU — Results & Final Demo

Responsible for:

- Final outputs and screenshots
- Result tables and statistics
- Sample input/output cases
- Final demo flow
- Presentation support
- Limitations and future scope

## System Workflow

```text
Input Image / Video / Webcam
            ↓
    Image Processing & I/O
            ↓
       AI / YOLO
            ↓
    Vehicle Detection
            ↓
   Vehicle Classification
            ↓
 Bounding Box + Confidence
            ↓
       Core Integration
            ↓
 Vehicle Counting / FPS
            ↓
    UI & Visualization
            ↓
     Testing & Analysis
            ↓
      Final Results



                    GITHUB
                       │
                       ▼
                    main
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   PINKU            PRITAM             RAJ
   AI/YOLO       Image Processing   Integration
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                   SANSKRITI
                       UI
                       │
                       ▼
                   SANTANU
              Testing & Performance
                       │
                       ▼
                  DEBANSHU
              Results & Final Demo
                       │
                       ▼
                    MAIN
                       │
                       ▼
             FINAL PROJECT DEMO