# export_pruned_yaml_from_pt.py
import torch
import yaml
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import (
    SPPF, C2PSA, A2C2f, C3k2
)
from ultralytics.nn.modules.vanillanet import vanillanetBlock
from ultralytics.nn.modules.head import Detect


PT_PATH = "/root/YOLOv12-new/runs/detect-vanilla/pruned_depgraph/weights/pruned_model.pt"
OUT_YAML = "pruned_from_pt_detect.yaml"
NC = 17  # 你的检测类别数（renshen.yaml 里是 17）




import torch.nn as nn



def get_conv_stride(x):
    """
    从 Conv / Conv2d / Sequential 中获取 stride
    """
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv as YOLOConv

    if isinstance(x, YOLOConv):
        return x.conv.stride[0]

    if isinstance(x, nn.Conv2d):
        return x.stride[0]

    if isinstance(x, nn.Sequential):
        for m in x:
            try:
                return get_conv_stride(m)
            except RuntimeError:
                continue

    raise RuntimeError(f"找不到 Conv2d 以获取 stride: {type(x)}")









def get_conv_channels(x):
    """
    从 Conv / Conv2d / Sequential 中安全获取 (in_channels, out_channels)
    """
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv as YOLOConv

    # 情况 1：YOLO 的 Conv 包装器
    if isinstance(x, YOLOConv):
        conv = x.conv
        return conv.in_channels, conv.out_channels

    # 情况 2：原生 Conv2d
    if isinstance(x, nn.Conv2d):
        return x.in_channels, x.out_channels

    # 情况 3：Sequential（递归找）
    if isinstance(x, nn.Sequential):
        for m in x:
            try:
                return get_conv_channels(m)
            except RuntimeError:
                continue

    raise RuntimeError(f"找不到 Conv2d: {type(x)}")






# ------------------------------------------------
# 1. 安全加载 pruned_model.pt
# ------------------------------------------------
torch.serialization.add_safe_globals([DetectionModel])
ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
model = ckpt["model"] if isinstance(ckpt, dict) else ckpt

assert isinstance(model, DetectionModel), "不是 DetectionModel，加载失败"

layers = model.model  # nn.ModuleList

# ------------------------------------------------
# 2. 构建 YAML 骨架
# ------------------------------------------------
yaml_dict = {
    "nc": NC,
    "scales": {
        "n": [1.0, 1.0, 1024]
    },
    "backbone": [],
    "head": []
}

# ------------------------------------------------
# 3. 逐层解析 nn.ModuleList → YAML
# ------------------------------------------------
def add(layer_list, f, n, m, args):
    layer_list.append([f, n, m, args])


for i, m in enumerate(layers):
    f = m.f if hasattr(m, "f") else -1
    n = m.n if hasattr(m, "n") else 1

    # -------- Conv --------
    if isinstance(m, Conv):
        c1, c2 = get_conv_channels(m.conv)
        k = m.conv.kernel_size[0]
        s = m.conv.stride[0]
        add(yaml_dict["backbone"], f, n, "Conv", [c1, c2, k, s])

    # -------- vanillanetBlock --------
    elif isinstance(m, vanillanetBlock):
        c1, _ = get_conv_channels(m.conv1)
        _, c2 = get_conv_channels(m.conv2)
        s = get_conv_stride(m.conv1)
        add(yaml_dict["backbone"], f, n, "vanillanetBlock", [c1, c2, 1, s])

    # -------- SPPF --------
    # elif isinstance(m, SPPF):
    #     c1, _ = get_conv_channels(m.cv1)
    #     _, c2 = get_conv_channels(m.cv2)
    #     k = m.k
    #     add(yaml_dict["backbone"], f, n, "SPPF", [c1, c2, k])
  

    elif isinstance(m, SPPF):
        # 输入输出通道
        c1, c2 = get_conv_channels(m.cv1)
    
        # kernel_size 从 MaxPool2d 里取
        if hasattr(m, "m") and hasattr(m.m, "kernel_size"):
            k = m.m.kernel_size
            if isinstance(k, tuple):
                k = k[0]
        else:
            # 极端兜底
            k = 5
    
        yaml_layers.append([
            from_idx,
            1,
            "SPPF",
            [c1, c2, k]
        ])


    # -------- C2PSA --------
    elif isinstance(m, C2PSA):
        c1 = m.cv1.conv.in_channels
        c2 = m.cv2.conv.out_channels
        add(yaml_dict["backbone"], f, n, "C2PSA", [c1, c2, 1])

    # -------- A2C2f --------
    elif isinstance(m, A2C2f):
        c1 = m.cv1.conv.in_channels
        c2 = m.cv2.conv.out_channels
        add(yaml_dict["head"], f, n, "A2C2f", [c1, c2, 1, False, -1])

    # -------- C3k2 --------
    elif isinstance(m, C3k2):
        c1 = m.cv1.conv.in_channels
        c2 = m.cv2.conv.out_channels
        add(yaml_dict["head"], f, n, "C3k2", [c1, c2, 1, True])

    # -------- Detect Head --------
    elif isinstance(m, Detect):
        ch = m.ch  # 每个检测层输入通道
        add(yaml_dict["head"], f, n, "Detect", [NC, ch])

    else:
        print(f"[WARN] 未识别层 {i}: {type(m)}，已跳过")

# ------------------------------------------------
# 4. 保存 YAML
# ------------------------------------------------
with open(OUT_YAML, "w") as f:
    yaml.safe_dump(yaml_dict, f, sort_keys=False)

print("✅ 剪枝后的 detect YAML 已导出：", OUT_YAML)
print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
