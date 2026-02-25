#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Sequence, Optional
import inspect

# =========================
# 0) 强制使用本项目 ultralytics（避免 /root/YOLO 抢占）
# =========================
ROOT = Path(__file__).resolve().parents[1]  # /root/YOLOv12-new

sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith("/root/YOLO"))]
sys.path.insert(0, str(ROOT))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

# =========================
# 1) imports
# =========================
import psutil
import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import AAttn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.vanillanet import vanillanetBlock, activation

# =========================
# 2) 路径配置
# =========================
MODEL_YAML = ROOT / "ultralytics/cfg/models/v12/yolo12-vanillanet-SPPF_MixPool.yaml"
WEIGHTS = Path("/root/YOLOv12-new/trained_models/yolov12n-vanillanet-SPPF_MixPool.pt")
DATA_YAML = Path("/root/YOLOv12-new/ultralytics/cfg/datasets/renshen.yaml")

# =========================
# 3) 训练/稀疏化超参
# =========================
TRAIN_IMGSZ = 512
BATCH = 4
WORKERS = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 5e-4
LOG_EVERY = 20

# DepGraph tracing 用更小分辨率（师兄脚本 dummy_size=64 很稳）
DG_IMGSZ = 64

# Group-SL 正则强度（先从 1e-4 起步）
REG = 1e-4

# 仅稀疏化训练，不做物理剪枝（pruning_ratio=0）
PRUNING_RATIO_FOR_SPARSE = 0.0


# =============================================================================
# A) 权重加载工具
# =============================================================================
def load_ultralytics_weights(model: nn.Module, weights_path: Path, device: str) -> None:
    assert weights_path.exists(), f"❌ 权重文件不存在: {weights_path}"
    ckpt = torch.load(str(weights_path), map_location=device)

    if isinstance(ckpt, dict):
        if ckpt.get("ema", None) is not None:
            state = ckpt["ema"].state_dict()
        elif ckpt.get("model", None) is not None:
            m = ckpt["model"]
            state = m.state_dict() if hasattr(m, "state_dict") else m
        else:
            state = ckpt
    else:
        state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✅ Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")


def batch_to_device(batch: dict, device: str) -> dict:
    img = batch["img"].to(device, non_blocking=True)
    img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
    batch["img"] = img
    for k, v in batch.items():
        if k == "img":
            continue
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


# =============================================================================
# B) 关键：Torch-Pruning 防爆补丁（完全照师兄的做法）
#    目的：防止 reshape/flatten/split 的 index_mapping 指数级膨胀 -> CPU 内存爆炸 -> Killed
# =============================================================================
def setup_torch_pruning() -> tp:
    import torch_pruning

    # 提高 DepGraph 递归深度上限（复杂嵌套时避免中断）
    try:
        import torch_pruning.dependency.constants as constants
    except Exception:
        try:
            constants = torch_pruning.dependency.constants
        except Exception:
            from torch_pruning.dependency import constants  # type: ignore

    old_depth = getattr(constants, "MAX_RECURSION_DEPTH", 1000)
    constants.MAX_RECURSION_DEPTH = 5000
    print(f"设置 MAX_RECURSION_DEPTH: {old_depth} -> {constants.MAX_RECURSION_DEPTH}")

    # 防止 flatten/split 的索引映射膨胀
    try:
        from torch_pruning.dependency import index_mapping as _im
        from torch_pruning import _helpers as _hp

        _orig_flatten_call = _im._FlattenIndexMapping.__call__
        _orig_split_call = _im._SplitIndexMapping.__call__

        def _safe_flatten_call(self, idxs):
            # 非 reverse 情况下强制近似恒等映射，避免 flatten 展开导致指数级增长
            if getattr(self, "reverse", False):
                return _orig_flatten_call(self, idxs)
            return [_hp._HybridIndex(idx=i.idx, root_idx=i.root_idx) for i in idxs]

        def _safe_split_call(self, idxs):
            try:
                return _orig_split_call(self, idxs)
            except Exception:
                off0 = self.offset[0] if isinstance(getattr(self, "offset", None), (list, tuple)) and len(self.offset) > 0 else 0
                off1 = self.offset[1] if isinstance(getattr(self, "offset", None), (list, tuple)) and len(self.offset) > 1 else off0
                if getattr(self, "reverse", False):
                    return [_hp._HybridIndex(idx=i.idx + off0, root_idx=i.root_idx) for i in idxs]
                span = max(off1 - off0, 1)
                return [_hp._HybridIndex(idx=max(min(i.idx - off0, span - 1), 0), root_idx=i.root_idx) for i in idxs]

        _im._FlattenIndexMapping.__call__ = _safe_flatten_call
        _im._SplitIndexMapping.__call__ = _safe_split_call
        print("✓ 已应用 index_mapping 防爆补丁（Flatten/Split）")
    except Exception as e:
        print(f"⚠️ 警告: index_mapping 防爆补丁失败: {e}")

    version = getattr(tp, "__version__", "unknown")
    print(f"torch-pruning 版本: {version}")
    return tp


# =============================================================================
# C) 自定义剪枝器（照师兄脚本做法）
# =============================================================================
# 让“块”作为根，避免 qkv 等卷积作为根导致依赖爆炸
ROOT_MODULE_TYPES = [AAttn, vanillanetBlock, activation, nn.Conv2d, nn.Linear]


def create_aattn_pruner(tp_mod: tp):
    class AAttnPruner(tp_mod.pruner.BasePruningFunc):
        TARGET_MODULES = (AAttn,)

        def prune_out_channels(self, layer: AAttn, idxs: Sequence[int]):
            idxs = list(idxs)
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, "conv") else proj
            tp_mod.pruner.function.prune_conv_out_channels(proj_conv, idxs)
            if hasattr(proj, "bn"):
                tp_mod.pruner.function.prune_batchnorm_out_channels(proj.bn, idxs)
            if hasattr(layer, "c2"):
                layer.c2 = proj_conv.out_channels
            return idxs

        def prune_in_channels(self, layer: AAttn, idxs: Sequence[int]):
            idxs = list(idxs)

            qkv = layer.qkv
            qkv_conv = qkv.conv if hasattr(qkv, "conv") else qkv
            old_C = qkv_conv.in_channels

            # 1) prune qkv input
            tp_mod.pruner.function.prune_conv_in_channels(qkv_conv, idxs)

            # 2) prune qkv output with 1:3 binding
            # 剪输入通道 i -> 输出要剪 i, i+old_C, i+2*old_C
            qkv_out_idxs = []
            for i in idxs:
                qkv_out_idxs.extend([i, i + old_C, i + 2 * old_C])

            tp_mod.pruner.function.prune_conv_out_channels(qkv_conv, qkv_out_idxs)
            if hasattr(qkv, "bn"):
                tp_mod.pruner.function.prune_batchnorm_out_channels(qkv.bn, qkv_out_idxs)

            # 3) prune PE depthwise (C->C, groups=C)
            if hasattr(layer, "pe"):
                pe = layer.pe
                pe_conv = pe.conv if hasattr(pe, "conv") else pe
                if isinstance(pe_conv, nn.Conv2d) and pe_conv.groups == pe_conv.in_channels:
                    keep = [i for i in range(pe_conv.in_channels) if i not in idxs]
                    pe_conv.weight = nn.Parameter(pe_conv.weight.data[keep].clone())
                    if pe_conv.bias is not None:
                        pe_conv.bias = nn.Parameter(pe_conv.bias.data[keep].clone())
                    pe_conv.in_channels = pe_conv.out_channels = pe_conv.groups = len(keep)
                    if hasattr(pe, "bn"):
                        bn = pe.bn
                        bn.weight = nn.Parameter(bn.weight.data[keep].clone())
                        bn.bias = nn.Parameter(bn.bias.data[keep].clone())
                        bn.running_mean = bn.running_mean[keep]
                        bn.running_var = bn.running_var[keep]
                        bn.num_features = len(keep)

            # 4) prune proj input
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, "conv") else proj
            tp_mod.pruner.function.prune_conv_in_channels(proj_conv, idxs)

            if hasattr(layer, "c1"):
                layer.c1 = qkv_conv.in_channels
            if hasattr(layer, "num_heads") and hasattr(layer, "head_dim"):
                layer.head_dim = qkv_conv.in_channels // layer.num_heads

            return idxs

        def get_out_channels(self, layer: AAttn):
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, "conv") else proj
            return proj_conv.out_channels

        def get_in_channels(self, layer: AAttn):
            qkv = layer.qkv
            qkv_conv = qkv.conv if hasattr(qkv, "conv") else qkv
            return qkv_conv.in_channels

    return AAttnPruner()


def create_vanillanet_block_pruner(tp_mod: tp):
    class VanillanetBlockPruner(tp_mod.pruner.BasePruningFunc):
        TARGET_MODULES = (vanillanetBlock,)

        def prune_out_channels(self, layer: vanillanetBlock, idxs: Sequence[int]):
            idxs = list(idxs)
            if layer.deploy:
                tp_mod.pruner.function.prune_conv_out_channels(layer.conv, idxs)
            else:
                conv2 = layer.conv2[0]
                bn2 = layer.conv2[1]
                tp_mod.pruner.function.prune_conv_out_channels(conv2, idxs)
                tp_mod.pruner.function.prune_batchnorm_out_channels(bn2, idxs)

            act = layer.act
            keep = [i for i in range(act.dim) if i not in idxs]
            new_dim = len(keep)

            act.weight = nn.Parameter(act.weight.data[keep].clone())
            if hasattr(act, "bn") and act.bn is not None:
                act.bn.weight = nn.Parameter(act.bn.weight.data[keep].clone())
                act.bn.bias = nn.Parameter(act.bn.bias.data[keep].clone())
                act.bn.running_mean = act.bn.running_mean[keep]
                act.bn.running_var = act.bn.running_var[keep]
                act.bn.num_features = new_dim
            if act.bias is not None:
                act.bias = nn.Parameter(act.bias.data[keep].clone())
            act.dim = new_dim
            return idxs

        def prune_in_channels(self, layer: vanillanetBlock, idxs: Sequence[int]):
            idxs = list(idxs)
            if layer.deploy:
                tp_mod.pruner.function.prune_conv_in_channels(layer.conv, idxs)
            else:
                conv1 = layer.conv1[0]
                bn1 = layer.conv1[1]
                tp_mod.pruner.function.prune_conv_in_channels(conv1, idxs)
                # conv1: dim->dim，所以 out 也同步剪
                tp_mod.pruner.function.prune_conv_out_channels(conv1, idxs)
                tp_mod.pruner.function.prune_batchnorm_out_channels(bn1, idxs)

                conv2 = layer.conv2[0]
                tp_mod.pruner.function.prune_conv_in_channels(conv2, idxs)
            return idxs

        def get_out_channels(self, layer: vanillanetBlock):
            return layer.act.dim

        def get_in_channels(self, layer: vanillanetBlock):
            if layer.deploy:
                return layer.conv.in_channels
            return layer.conv1[0].in_channels

    return VanillanetBlockPruner()


def create_activation_pruner(tp_mod: tp):
    class ActivationPruner(tp_mod.pruner.BasePruningFunc):
        TARGET_MODULES = (activation,)

        def prune_out_channels(self, layer: activation, idxs: Sequence[int]):
            idxs = list(idxs)
            keep = [i for i in range(layer.dim) if i not in idxs]
            new_dim = len(keep)

            layer.weight = nn.Parameter(layer.weight.data[keep].clone())
            if hasattr(layer, "bn") and layer.bn is not None:
                layer.bn.weight = nn.Parameter(layer.bn.weight.data[keep].clone())
                layer.bn.bias = nn.Parameter(layer.bn.bias.data[keep].clone())
                layer.bn.running_mean = layer.bn.running_mean[keep]
                layer.bn.running_var = layer.bn.running_var[keep]
                layer.bn.num_features = new_dim
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.data[keep].clone())

            layer.dim = new_dim
            return idxs

        def prune_in_channels(self, layer: activation, idxs: Sequence[int]):
            # groups=dim 的分组卷积：in/out 同步
            return self.prune_out_channels(layer, idxs)

        def get_out_channels(self, layer: activation):
            return layer.dim

        def get_in_channels(self, layer: activation):
            return layer.dim

    return ActivationPruner()



def args_to_dict(a):
    # Ultralytics 读取 ckpt["train_args"] 需要 dict
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    # SimpleNamespace / argparse.Namespace / 任何带 __dict__ 的对象
    if hasattr(a, "__dict__"):
        return dict(vars(a))
    return {}







def save_yolo_ckpt(model: DetectionModel, save_path: Path, model_yaml: Path, epoch: int = -1):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": deepcopy(model).cpu().half(),  # DetectionModel，给 YOLO() 直接加载
        "ema": None,
        "optimizer": None,
        "train_args": args_to_dict(getattr(model, "args", None)),  # ✅ dict
        "epoch": epoch,
        "yaml": str(model_yaml),
        "date": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    torch.save(ckpt, str(save_path))





# =============================================================================
# D) ignored layers（照师兄逻辑）
# =============================================================================
def _unwrap_conv(m: nn.Module) -> Optional[nn.Conv2d]:
    if isinstance(m, nn.Conv2d):
        return m
    if isinstance(m, Conv) and hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
        return m.conv
    return None


def _last_conv_in_branch(branch: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for mm in branch.modules():
        if isinstance(mm, (nn.Conv2d, Conv)):
            last = mm
    return _unwrap_conv(last) if last is not None else None


def get_ignored_layers(model: nn.Module) -> List[nn.Module]:
    ignored: List[nn.Module] = []

    # 1) DFL 固定结构不剪
    for m in model.modules():
        if m.__class__.__name__ == "DFL":
            ignored.append(m)

    # 2) AAttn 的 qkv.conv 交给自定义 pruner（避免 TP 追 qkv 产生爆炸依赖）
    for m in model.modules():
        if isinstance(m, AAttn):
            qkv_conv = getattr(getattr(m, "qkv", None), "conv", None)
            if isinstance(qkv_conv, nn.Conv2d):
                ignored.append(qkv_conv)

    # 3) 只忽略 Detect 的最终输出卷积（每尺度 reg + cls 各一个）
    detect: Optional[Detect] = None
    for sub in model.modules():
        if isinstance(sub, Detect):
            detect = sub
            break

    if detect is not None:
        for branch in list(detect.cv2) + list(detect.cv3):
            last_conv = _last_conv_in_branch(branch)
            if last_conv is not None:
                ignored.append(last_conv)

    print(f"忽略的层数: {len(ignored)}")
    return ignored


# =============================================================================
# E) 构建 Pruner（用于 sparse regularize）
# =============================================================================
def build_pruner_for_sparse(model: nn.Module, example_inputs: torch.Tensor, reg: float) -> tp.pruner.BasePruner:
    imp = tp.importance.BNScaleImportance()

    customized_pruners = {
        AAttn: create_aattn_pruner(tp),
        vanillanetBlock: create_vanillanet_block_pruner(tp),
        activation: create_activation_pruner(tp),
    }

    ignored_layers = get_ignored_layers(model)

    # 稀疏化阶段：pruning_ratio=0.0，只用 reg 做 regularize
    # global_pruning 开不开都行；为保持一致性，这里用 True（和师兄一致）
    pruner = tp.pruner.BNScalePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=PRUNING_RATIO_FOR_SPARSE,
        global_pruning=True,
        ignored_layers=ignored_layers,
        root_module_types=ROOT_MODULE_TYPES,
        customized_pruners=customized_pruners,
        reg=reg,
    )
    return pruner


# =============================================================================
# F) main
# =============================================================================
def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 关键：应用 torch-pruning 防爆补丁
    setup_torch_pruning()

    # ---------- 1) 构建模型 ----------
    assert MODEL_YAML.exists(), f"❌ 找不到模型yaml: {MODEL_YAML}"
    model: DetectionModel = DetectionModel(str(MODEL_YAML))

    # ---------- 2) 加载 baseline 权重 ----------
    assert WEIGHTS.exists(), f"❌ 找不到权重: {WEIGHTS}"
    load_ultralytics_weights(model, WEIGHTS, device="cpu")

    # loss 初始化需要 args
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        label_smoothing=0.0, fl_gamma=0.0, imgsz=TRAIN_IMGSZ
    )

    # ---------- 3) 构建 DepGraph + pruner（CPU 上做，最稳） ----------
    print("Building DepGraph / Pruner for sparse (BNScalePruner + customized_pruners)...")

    # 让 TP 能追踪
    for p in model.parameters():
        p.requires_grad_(True)

    model = model.cpu()
    model.eval()

    example_inputs = torch.randn(1, 3, DG_IMGSZ, DG_IMGSZ)

    proc = psutil.Process(os.getpid())
    print("RSS(GB) before pruner:", proc.memory_info().rss / 1024**3)

    t0 = time.time()
    AAttn.set_trace_mode(True)
    pruner = build_pruner_for_sparse(model, example_inputs, reg=REG)
    AAttn.set_trace_mode(False)

    print("RSS(GB) after pruner:", proc.memory_info().rss / 1024**3)
    print(f"✅ Pruner ready. time={time.time() - t0:.1f}s")

    # ---------- 4) 数据加载 ----------
    print("Building dataloader...")
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(DATA_YAML))
    model.nc = int(data["nc"])

    from ultralytics.cfg import get_cfg
    cfg = get_cfg(overrides={
        "imgsz": TRAIN_IMGSZ,
        "task": "detect",
        "rect": False,
        "cache": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
    })

    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    train_path = str(data["train"])
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=train_path,
        batch=BATCH,
        data=data,
        mode="train",
        rect=False,
        stride=32
    )
    train_loader = build_dataloader(
        dataset=dataset,
        batch=BATCH,
        workers=WORKERS,
        shuffle=True,
        rank=-1,
    )

    # ---------- 5) 移到 GPU 训练 ----------
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("Start sparse training (DepGraph regularize)...")
    global_step = 0

    for epoch in range(EPOCHS):
        # 每 epoch 更新 regularizer（TP 推荐）
        if hasattr(pruner, "update_regularizer"):
            try:
                pruner.update_regularizer()
            except TypeError:
                pruner.update_regularizer(model)

        for i, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            # Ultralytics detection loss
            det_loss = model.loss(batch)[0]
            det_loss.backward()

            # 稀疏正则（注意：不要传 reg=，reg 已在 pruner 构造时传入）
            pruner.regularize(model)

            optimizer.step()

            if global_step % LOG_EVERY == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0.0
                rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                print(f"[E{epoch+1}/{EPOCHS}][{i+1}] det={det_loss.item():.4f} "
                      f"gpu_mem={gpu_mem:.2f}GB rss={rss:.2f}GB")
            global_step += 1



        save_dir = ROOT / "sparsed_models" / "yolov12n-vanillanet-SPPF_MixPool"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # A) 研究用 raw（torch.load 直接拿到 model / 便于你 debug/pruning）

        raw_path = save_dir / f"sparse_epoch{epoch+1}_raw_state.pt"
        torch.save({
            "state_dict": deepcopy(model).cpu().state_dict(),
            "train_args": args_to_dict(getattr(model, "args", None)),
            "yaml": str(MODEL_YAML),
            "epoch": epoch+1,
        }, str(raw_path))

        
        # B) YOLO 格式（给 YOLO().val/predict 直接用）
        yolo_path = save_dir / f"sparse_epoch{epoch+1}.pt"
        save_yolo_ckpt(model, yolo_path, MODEL_YAML,epoch=epoch+1)
        
        print(f"✅ Saved raw:  {raw_path}")
        print(f"✅ Saved yolo: {yolo_path}")

    raw_final = save_dir / "sparse_final_raw_state.pt"
    torch.save({
        "state_dict": deepcopy(model).cpu().state_dict(),
        "train_args": args_to_dict(getattr(model, "args", None)),
        "yaml": str(MODEL_YAML),
        "epoch": EPOCHS,
    }, str(raw_final))

    
    torch.save({"model": deepcopy(model).cpu(), "train_args":args_to_dict(getattr(model, "args", None))}, str(raw_final))
    yolo_final = save_dir / "sparse_final.pt"
    save_yolo_ckpt(model, yolo_final, MODEL_YAML,epoch=EPOCHS)
    
    print(f"✅ Done. raw:  {raw_final}")
    print(f"✅ Done. yolo: {yolo_final}")




if __name__ == "__main__":
    main()
