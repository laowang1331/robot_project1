"""Train a YOLO object-detection model with Ultralytics.

Install once in the PyCharm interpreter terminal:
    python -m pip install ultralytics

Before running, edit data.yaml so its paths and class names match your dataset.
"""

from pathlib import Path

from ultralytics import YOLO


# Change these values for your project.
DATA_YAML = Path(__file__).with_name("data.yaml")
MODEL_WEIGHTS = "yolo11n.pt"  # Small pretrained model; a good starting point.
EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 8  # Reduce this if an out-of-memory error occurs.
DEVICE = 0  # Use GPU 0. Change to "cpu" when no NVIDIA CUDA GPU is available.


def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration was not found: {DATA_YAML}")

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        amp = False,
        device=DEVICE,
        project="runs/detect",
        name="mouse_bottle",
        exist_ok=True,
        patience=20,
    )
    print("Training completed.")
    print("Best model:", Path(results.save_dir) / "weights" / "best.pt")

if __name__ == "__main__":
    main()
