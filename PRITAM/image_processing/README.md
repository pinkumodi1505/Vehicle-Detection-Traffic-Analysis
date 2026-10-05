# PRITAM — Image Processing & I/O

## Role

Image Processing & I/O Developer

## Responsibility

Responsible for preparing input images and video data for the vehicle detection system and handling input/output operations.

## Responsibilities

- Implement image preprocessing.
- Resize images and video frames when required.
- Apply brightness and contrast adjustments where required.
- Apply noise reduction where useful.
- Extract frames from videos.
- Validate image and video inputs.
- Handle input file formats.
- Save processed images, videos, and detection outputs.
- Provide properly processed inputs to the AI / YOLO module.

## Module Output

- Preprocessed images
- Processed video frames
- Extracted video frames
- Validated input data
- Saved processed outputs

## Integration

The Image Processing & I/O module prepares the input before it is passed to the AI / YOLO detection module.

It also handles saving the processed results produced by the application.

## Technology

- Python
- OpenCV
- NumPy
- PIL/Pillow