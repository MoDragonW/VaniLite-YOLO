from ultralytics import YOLO
from exporter import Exporter

# 你的 yolov12n 权重路径
pt_path = "/root/yolov12-main/runs/segment/train18/weights/best.pt"   # 换成实际路径

# 加载模型并拿到 nn.Module
yolo = YOLO(pt_path)
model = yolo.model
# 给模型补一个 pt_path，方便导出器写元数据与默认文件名
setattr(model, "pt_path", pt_path)

# 配置导出参数：ONNX + 动态尺寸 + 图优化 + 指定 opset
exp = Exporter(overrides=dict(
    format="onnx",
    imgsz=(640, 640),   # 按你训练/推理尺寸
    dynamic=False,       # 需要动态输入就开；开了会在 CPU 上导出
    simplify=True,      # 精简图
    opset=12,           # 不确定就用 17/19 之一
    batch=1
))

onnx_file = exp(model=model)  # 执行导出
print("导出完成：", onnx_file)
