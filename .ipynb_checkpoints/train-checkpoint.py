from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolo12-seg.yaml")  # build a new model from scratch
#model.load("yolov12n-seg.pt")

# Train the model
results = model.train(data="toug-coco128-seg.yaml")