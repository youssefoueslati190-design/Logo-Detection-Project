from ultralytics import YOLO
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent

    # Use your exact folder names
    model1_path = project_root / "results" / "model1_yolov8_baseline8" / "weights" / "best.pt"
    model2_path = project_root / "results" / "model2_yolov8_medium6" / "weights" / "best.pt"

    demo_folder = project_root / "demo_images"

    model1 = YOLO(str(model1_path))
    model2 = YOLO(str(model2_path))

    # Common prediction settings
    common_args = dict(
        source=str(demo_folder),
        imgsz=640,
        conf=0.6,
        device="cpu",
        save=True,
        save_conf=True,
        project=str(project_root / "results"),
        exist_ok=True
    )

    # Run predictions for Model 1
    model1.predict(name="pred_model1", **common_args)

    # Run predictions for Model 2
    model2.predict(name="pred_model2", **common_args)

    print("Predictions finished! Check results/pred_model1 and results/pred_model2")

if __name__ == "__main__":
    main()





