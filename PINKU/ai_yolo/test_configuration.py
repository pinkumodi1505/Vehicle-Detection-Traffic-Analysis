from vehicle_detector import VehicleDetector


def main():
    detector = VehicleDetector()

    print("PINKU YOLO Configuration")
    print("-" * 40)
    print("Model:", detector.model.ckpt_path)
    print("Confidence:", detector.confidence)
    print("Image size:", detector.image_size)
    print("IoU:", detector.iou)
    print("Vehicle classes:", detector.VEHICLE_CLASSES)


if __name__ == "__main__":
    main()