import os
from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Get absolute path to data.yaml to avoid missing file errors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, "dataset", "data.yaml")

    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        name="helmet_custom_model"
    )

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
