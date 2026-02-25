from ultralytics import YOLO
# yolo11/yolo11n.pt
model = YOLO('/root/yolov12-main/runs/segment/train18/weights/best.pt')
results = model.val(data="/root/autodl-tmp/data.yaml",  imgsz=640)