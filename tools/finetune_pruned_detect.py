#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional

# =========================
# 0) 
# =========================
ROOT = Path(__file__).resolve().parents[1]  # /root/YOLOv12-new
sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith("/root/YOLO"))]
sys.path.insert(0, str(ROOT))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

# =========================
# 1)
# =========================
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset, build_dataloader

# =========================
# 2) 配置区
# =========================
PRUNED_CKPT = ROOT / "trained_models/yolov12n-vanillanet-pruned-sparsity0.25.pt"
DATA_YAML = ROOT / "ultralytics/cfg/datasets/renshen.yaml"

SAVE_DIR = ROOT / "finetuned_models" / "yolov12n-vanillanet-pruned-sparsity0.25"
EPOCHS = 120
IMGSZ = 512
BATCH = 16
WORKERS = 4
DEVICE = "cuda:0"

# 训练超参
LR = 5e-4
WEIGHT_DECAY = 5e-4
LOG_EVERY = 50

# 是否开启 AMP
USE_AMP = True

# 每多少个 epoch 做一次评估并打印 mAP
EVAL_EVERY = 1

# 用 val 还是 test（你的 yaml 若没有 test，就自动回退到 val）
EVAL_SPLIT = "val"   # "val" 或 "test"
# =========================


def args_to_dict(a):
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    if hasattr(a, "__dict__"):
        return dict(vars(a))
    return {}


def fix_model_args_and_criterion(model: nn.Module, imgsz: int):
    """
    1) 把 model.args 修成 SimpleNamespace，保证 loss 里用 .box/.cls/.dfl 不会炸
    2) 删除旧 criterion，避免其中的张量停留在 CPU 或 hyp 类型不对
    """
    a = getattr(model, "args", {})
    d = a if isinstance(a, dict) else (vars(a) if hasattr(a, "__dict__") else {})

    model.args = SimpleNamespace(
        box=float(d.get("box", 7.5)),
        cls=float(d.get("cls", 0.5)),
        dfl=float(d.get("dfl", 1.5)),
        label_smoothing=float(d.get("label_smoothing", 0.0)),
        fl_gamma=float(d.get("fl_gamma", 0.0)),
        imgsz=int(d.get("imgsz", imgsz)),
    )

    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    if hasattr(model, "loss_fn"):
        delattr(model, "loss_fn")


def save_yolo_ckpt(model: nn.Module, save_path: Path, train_args: dict, extra: Optional[Dict[str, Any]] = None):
    """
    保存为 YOLO 可直接加载的 ckpt(dict)，包含 DetectionModel 对象。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": deepcopy(model).cpu().half(),
        "ema": None,
        "optimizer": None,
        "train_args": dict(train_args),
        "yaml": str(getattr(model, "yaml_file", "")),
        "date": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(save_path))


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


def extract_map(metrics) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    尽可能鲁棒地从 Ultralytics 的 val() 返回对象里提取 mAP50-95 和 mAP50。
    返回: (map50_95, map50, raw_dict)
    """
    raw = {}

    # 1) 结果可能有 results_dict
    if hasattr(metrics, "results_dict"):
        raw = dict(getattr(metrics, "results_dict", {}) or {})
    elif isinstance(metrics, dict):
        raw = dict(metrics)
    else:
        # 2) 尝试从对象属性里抠
        for attr in ["map", "map50"]:
            if hasattr(metrics, attr):
                raw[attr] = getattr(metrics, attr)

    # 常见 key 兼容
    candidates_map = [
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "map", "mAP50-95", "box/map"
    ]
    candidates_map50 = [
        "metrics/mAP50(B)", "metrics/mAP50", "map50", "mAP50", "box/map50"
    ]

    map5095 = None
    map50 = None
    for k in candidates_map:
        if k in raw and isinstance(raw[k], (int, float)):
            map5095 = float(raw[k])
            break
    for k in candidates_map50:
        if k in raw and isinstance(raw[k], (int, float)):
            map50 = float(raw[k])
            break

    return map5095, map50, raw



@torch.no_grad()
def evaluate_epoch(model: nn.Module, data_yaml: str, imgsz: int, batch: int, device: str, split: str) -> Tuple[Optional[float], Optional[float]]:
    """
    用训练模型的 deepcopy 做评估，避免 Ultralytics val() 的 inference_mode 污染训练模型。
    """
    # 1) 拷贝一个评估用模型
    eval_model = deepcopy(model).to(device).eval()

    # 2) 修复 args/criterion（评估模型也需要）
    fix_model_args_and_criterion(eval_model, imgsz)

    # 3) split 若不存在回退
    ds = check_det_dataset(data_yaml)
    if split not in ds:
        split = "val"

    # 4) 用 YOLO 只当“壳”来跑 val pipeline
    y = YOLO(PRUNED_CKPT)
    y.model = eval_model

    metrics = y.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        plots=False,
        save=False,
        verbose=False,
    )

    map5095, map50, _ = extract_map(metrics)

    # 5) 释放显存
    del eval_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return map5095, map50







def main():
    # 速度相关（可选）
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "weights").mkdir(parents=True, exist_ok=True)

    # 1) 加载剪枝模型（结构已改变）
    yolo = YOLO(PRUNED_CKPT)
    model = yolo.model
    model.to(DEVICE)

    # 关键修复：args + criterion
    fix_model_args_and_criterion(model, IMGSZ)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Debug] Loaded pruned model params = {n_params:,}")

    # 2) 构建数据（train loader）
    data = check_det_dataset(DATA_YAML)
    model.nc = int(data["nc"])

    cfg = get_cfg(overrides={
        "imgsz": IMGSZ,
        "task": "detect",
        "rect": False,
        "cache": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
    })

    train_dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=str(data["train"]),
        batch=BATCH,
        data=data,
        mode="train",
        rect=False,
        stride=32,
    )
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch=BATCH,
        workers=WORKERS,
        shuffle=True,
        rank=-1,
    )

    # 3) 优化器 & AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 4) 训练 args（用于保存）
    train_args = args_to_dict(getattr(model, "args", None))
    train_args.setdefault("task", "detect")
    train_args["imgsz"] = IMGSZ

    best_map = -1.0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # 每个 epoch 开头，确保 criterion 在正确设备、hyp 类型正确
        fix_model_args_and_criterion(model, IMGSZ)

        for i, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
                loss = model.loss(batch)[0]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % LOG_EVERY == 0:
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"[E{epoch}/{EPOCHS}][{i}] loss={loss.item():.4f} gpu_mem={mem:.2f}GB")
            global_step += 1

        # --- 每个 epoch 保存 last ---
        last_path = SAVE_DIR / "weights" / "last.pt"
        save_yolo_ckpt(model, last_path, train_args, extra={"epoch": epoch})
        print(f"[Epoch {epoch}] Saved last: {last_path}")

        # --- 每个 epoch 评估并打印 mAP ---
        if epoch % EVAL_EVERY == 0:
            model.eval()
            map5095, map50 = evaluate_epoch(model, DATA_YAML, IMGSZ, BATCH, DEVICE, EVAL_SPLIT)
            print(f"[Epoch {epoch}] {EVAL_SPLIT} mAP50-95={map5095} mAP50={map50}")

            # --- 保存 best.pt（以 mAP50-95 为准）---
            if map5095 is not None and map5095 > best_map:
                best_map = map5095
                best_path = SAVE_DIR / "weights" / "best.pt"
                save_yolo_ckpt(model, best_path, train_args, extra={"epoch": epoch, "best_map": best_map})
                print(f"[Epoch {epoch}] ✅ New BEST! mAP50-95={best_map:.4f} saved: {best_path}")


    print("✅ Training done.")
    print(f"Best mAP50-95={best_map:.4f}")
    print(f"Outputs: {SAVE_DIR}/weights/best.pt and last.pt")


if __name__ == "__main__":
    main()
