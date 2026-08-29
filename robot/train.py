from pathlib import Path
from ultralytics import YOLO


DATA = r"E:\python\robot\data.yaml"
ROOT = Path(r"E:\python\robot")

i = 1
while (ROOT / f"train_{i}").exists():
    i += 1

name = f"train_{i}"
model = YOLO("yolo11n.pt")
model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    project=str(ROOT),#表示把训练好的结果放入ROOT指定的文件夹
    name=name,#表示将结果放入第i轮训练文件夹中
    exist_ok=False,
)

print(f"训练完成：{ROOT / name / 'weights' / 'best.pt'}")