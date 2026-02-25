# diagnose_model_nc.py
import torch
from ultralytics import YOLO
import sys

model_path = 'runs/seg-vanilla/pruned_depgraph/weights/pruned_model.pt'
print(f"检查模型: {model_path}")

# 尝试1: 作为完整YOLO对象加载
try:
    yolo = YOLO(model_path)
    print(f"[YOLO对象] 任务: {yolo.task}")
    if hasattr(yolo, 'model'):
        print(f"[YOLO对象] 模型类: {type(yolo.model)}")
        # 尝试获取nc
        if hasattr(yolo.model, 'nc'):
            print(f"[YOLO对象] yolo.model.nc = {yolo.model.nc}")
        # 深度搜索 Detect 模块
        for name, m in yolo.model.named_modules():
            if 'Detect' in m.__class__.__name__:
                print(f"[YOLO对象] 找到模块: {name}, type: {type(m)}")
                if hasattr(m, 'nc'):
                    print(f"  -> 该模块的 nc = {m.nc}")
                if hasattr(m, 'cv2') and m.cv2:
                    print(f"  -> cv2 输出通道: {[layer.out_channels for layer in m.cv2]}")
                if hasattr(m, 'cv3') and m.cv3:
                    print(f"  -> cv3 输出通道: {[layer.out_channels for layer in m.cv3]}")
except Exception as e:
    print(f"[YOLO对象] 加载或检查失败: {e}")

print("\n" + "="*60 + "\n")

# 尝试2: 直接加载torch checkpoint，查看原始数据
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print("[Torch Checkpoint] 类型:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("[Torch Checkpoint] 字典键:", checkpoint.keys())
        if 'model' in checkpoint:
            model_raw = checkpoint['model']
            print(f"[Torch Checkpoint] 'model' 字段类型: {type(model_raw)}")
            # 尝试寻找 model.yaml 或 args
            for key in ['model', 'args', 'cfg']:
                if key in checkpoint:
                    if isinstance(checkpoint[key], dict) and 'nc' in checkpoint[key]:
                        print(f"[Torch Checkpoint] 在['{key}']中找到 nc = {checkpoint[key]['nc']}")
except Exception as e:
    print(f"[Torch Checkpoint] 加载失败: {e}")