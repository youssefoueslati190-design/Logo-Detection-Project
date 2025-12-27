from ultralytics import YOLO
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent

    data_yaml = project_root / "data" / "logos" / "data.yaml"

    print("Using dataset yaml:", data_yaml)
    print("Exists? ->", data_yaml.is_file())

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=60,
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(project_root / "results"),
        name="model1_yolov8_baseline",
        augment=True,
    )


if __name__ == "__main__":
    main()



