from ultralytics import YOLO
from pathlib import Path

def main():
    # 1) Project root
    project_root = Path(__file__).resolve().parent.parent

    # 2) Correct absolute path to your dataset YAML
    data_yaml = project_root / "data" / "logos" / "data.yaml"

    # Debug print
    print("Using dataset yaml:", data_yaml)
    print("Exists? ->", data_yaml.is_file())

    # 3) Load YOLOv8 Medium (Model 2)
    model = YOLO("yolov8m.pt")

    # 4) Train Model 2
    model.train(
        data=str(data_yaml),
        epochs=70,
        imgsz=768,
        batch=8,
        device="cpu",
        project=str(project_root / "results"),
        name="model2_yolov8_medium",
        augment=True,
    )

if __name__ == "__main__":
    main()





