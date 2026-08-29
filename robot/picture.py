

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def export_dataset_images(
    video_path: Path,
    dataset_dir: Path,
    samples_per_second: float = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[int, int]:
    """Sample a video and return (train_count, val_count)."""
    if not 0 < samples_per_second:
        raise ValueError("samples_per_second must be greater than 0")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Cannot determine the video's frame rate")

    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Generate a reproducible 80/20 distribution independently of frame order.
    chooser = random.Random(seed)
    frames_between_samples = max(1, round(fps / samples_per_second))
    prefix = video_path.stem
    frame_index = 0
    image_index = 0
    train_count = 0
    val_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frames_between_samples == 0:
                target_dir = train_dir if chooser.random() < train_ratio else val_dir
                target_path = target_dir / f"{prefix}_{image_index:06d}.jpg"
                if not cv2.imwrite(str(target_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Failed to save image: {target_path}")

                if target_dir == train_dir:
                    train_count += 1
                else:
                    val_count += 1
                image_index += 1

            frame_index += 1
    finally:
        capture.release()

    return train_count, val_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export video frames into YOLO train/val image folders.")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("dataset", type=Path, help="YOLO dataset folder")
    parser.add_argument("--samples-per-second", type=float, default=2, help="Number of exported images each second")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction for images/train (default: 0.8)")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video file does not exist: {args.video}")

    train_count, val_count = export_dataset_images(
        args.video,
        args.dataset,
        args.samples_per_second,
        args.train_ratio,
    )
    print(f"Created {train_count} training images and {val_count} validation images.")
    print(f"Images are in: {args.dataset.resolve() / 'images'}")


if __name__ == "__main__":
    train_count, val_count = export_dataset_images(
        video_path=Path(r"E:\background.mp4"),
        dataset_dir=Path(r"C:\Users\dell\PycharmProjects\PythonProject"),
        samples_per_second=4,
        train_ratio=0.8,
    )
    print(f"训练集: {train_count} 张，验证集: {val_count} 张")
