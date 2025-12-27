from ultralytics import YOLO
from pathlib import Path

def eval_model(name: str, weights_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "data" / "logos" / "data.yaml"

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",      # or "val" if you prefer
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(project_root / "results"),
        name=f"eval_{name}",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n{name} results:")
    print(f"  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")

def main():
    project_root = Path(__file__).resolve().parent.parent

    # Same folders as in predict_logos.py
    model1_weights = project_root / "results" / "model1_yolov8_baseline8" / "weights" / "best.pt"
    model2_weights = project_root / "results" / "model2_yolov8_medium6" / "weights" / "best.pt"

    eval_model("model1_yolov8_baseline8", model1_weights)
    eval_model("model2_yolov8_medium6", model2_weights)

if __name__ == "__main__":
    main()



