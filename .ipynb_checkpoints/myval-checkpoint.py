import sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[0]  # 你的项目根自己按实际改
sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith("/root/YOLO"))]
sys.path.insert(0, str(ROOT))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]



from ultralytics import YOLO

# model_names = [
#     '/root/YOLO/results/yolo11n/weights/best.pt','/root/YOLO/results/yolo12n/weights/best.pt',
#     '/root/YOLO/results/yolov10n/weights/best.pt','/root/YOLO/results/yolov5n/weights/best.pt','/root/YOLO/results/yolov8n/weights/best.pt'
# ]

# model_names = [
    
# ]
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' 


# model = YOLO(r'/root/YOLOv12-new/runs/detect-vanilla/pruned_depgraph/weights/pruned_yolo.pt')

# model = YOLO(r'/root/YOLOv12-new/results/yolo12-vanillanet-SPPF_MixPool/weights/best.pt')
# model = YOLO(r'/root/YOLOv12-new/trained_models/yolov11.pt')
# model = YOLO(r'/root/YOLOv12-new/sparsed_models/yolov12n-vanillanet-SPPF_MixPool/sparse_final.pt')
# model = YOLO(r'/root/YOLOv12-new/pruned_models/pruned_detect_depgraph/yolov12n-vanillanet-SPPF_MixPool.pt')
model = YOLO(r'/root/YOLOv12-new/finetuned_models/yolov12n-vanillanet-SPPF_MixPool/weights/best.pt')



model.val(
    split='test',
    data='renshen.yaml'
   
)
 # data='renshen.yaml'