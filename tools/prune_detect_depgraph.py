#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, json, math, time
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import torch
import torch.nn as nn

# =========================
# 0) 强制使用本项目 ultralytics（避免 /root/YOLO 抢占）
# =========================
ROOT = Path(__file__).resolve().parents[1]  # /root/YOLOv12-new
sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith("/root/YOLO"))]
sys.path.insert(0, str(ROOT))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]
# ====== 固定配置 ======
SPARSE_CKPT = ROOT / "trained_models/yolov12n-vanillanet-sparsed.pt"
SPARSITY = 0.25
DUMMY = 64
DEVICE = "cpu"  # 推荐先 cpu；如果要 cuda 就改成 "cuda:0"
SAVE_DIR = ROOT / "pruned_models/pruned_detect_depgraph"


import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.vanillanet import vanillanetBlock, activation

from ultralytics.nn.modules.block import Attention
# =============== 可改配置 ===============
MODEL_YAML = ROOT / "ultralytics/cfg/models/v12/yolo12-vanillanet-SPPF_MixPool.yaml"
DEFAULT_DUMMY = 64  # DepGraph tracing 输入尺寸（64 最稳）
# ======================================


# -------------------------
# Torch-Pruning 防爆补丁（照你师兄逻辑）
# -------------------------
def setup_torch_pruning():
    import torch_pruning

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

    # 防止 reshape/flatten 的索引映射指数级膨胀
    try:
        from torch_pruning.dependency import index_mapping as _im
        from torch_pruning import _helpers as _hp

        _orig_flatten_call = _im._FlattenIndexMapping.__call__
        _orig_split_call = _im._SplitIndexMapping.__call__

        def _safe_flatten_call(self, idxs):
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

    print("torch-pruning 版本:", getattr(tp, "__version__", "unknown"))
    return tp


# -------------------------
# root_module_types（照你师兄）
# -------------------------
ROOT_MODULE_TYPES = [AAttn, vanillanetBlock, activation, nn.Conv2d, nn.Linear]


# -------------------------
# 自定义剪枝器：AAttn / vanillanetBlock / activation
# （照你师兄的逻辑，保证 QKV 1:3 对齐）
# -------------------------
def create_aattn_pruner(tp_mod):
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

            # 2) prune qkv output with 1:3 binding: i, i+old_C, i+2*old_C
            qkv_out_idxs = []
            for i in idxs:
                qkv_out_idxs.extend([i, i + old_C, i + 2 * old_C])

            tp_mod.pruner.function.prune_conv_out_channels(qkv_conv, qkv_out_idxs)
            if hasattr(qkv, "bn"):
                tp_mod.pruner.function.prune_batchnorm_out_channels(qkv.bn, qkv_out_idxs)

            # 3) prune PE depthwise
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


def create_vanillanet_block_pruner(tp_mod):
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
                tp_mod.pruner.function.prune_conv_out_channels(conv1, idxs)
                tp_mod.pruner.function.prune_batchnorm_out_channels(bn1, idxs)

                conv2 = layer.conv2[0]
                tp_mod.pruner.function.prune_conv_in_channels(conv2, idxs)
            return idxs

        def get_out_channels(self, layer: vanillanetBlock):
            return layer.act.dim

        def get_in_channels(self, layer: vanillanetBlock):
            return layer.conv.in_channels if layer.deploy else layer.conv1[0].in_channels

    return VanillanetBlockPruner()


def create_activation_pruner(tp_mod):
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
            return self.prune_out_channels(layer, idxs)

        def get_out_channels(self, layer: activation):
            return layer.dim

        def get_in_channels(self, layer: activation):
            return layer.dim

    return ActivationPruner()


# -------------------------
# ignored_layers（照你师兄）：DFL + AAttn.qkv.conv + Detect 最后输出 conv
# -------------------------
def _unwrap_conv(m):
    if isinstance(m, nn.Conv2d):
        return m
    if isinstance(m, Conv) and hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
        return m.conv
    return None

def _last_conv_in_branch(branch):
    last = None
    for mm in branch.modules():
        if isinstance(mm, (nn.Conv2d, Conv)):
            last = mm
    return _unwrap_conv(last) if last is not None else None

def _ignore_conv_wrapper_or_conv(ignored, m):
    # m 可能是 Conv wrapper，也可能是 nn.Conv2d
    if isinstance(m, Conv):
        ignored.append(m)
        if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
            ignored.append(m.conv)
        if hasattr(m, "bn") and m.bn is not None:
            ignored.append(m.bn)
    elif isinstance(m, nn.Conv2d):
        ignored.append(m)

def get_ignored_layers(model: nn.Module):
    ignored = []
    # ... 你原来的 DFL / AAttn.qkv / Detect last conv 逻辑 ...

    # 4) 关键：保护 PSA 的 Attention，避免 qkv 输出通道被剪坏导致 view 崩溃
    for m in model.modules():
        if isinstance(m, Attention):
            # 保护 qkv/proj/pe 三条支路
            _ignore_conv_wrapper_or_conv(ignored, m.qkv)
            _ignore_conv_wrapper_or_conv(ignored, m.proj)
            _ignore_conv_wrapper_or_conv(ignored, m.pe)
            ignored.append(m)  # 也可以把模块本体加入 ignored

    # 去重（很重要，否则 ignored 里会很多重复对象）
    seen = set()
    uniq = []
    for x in ignored:
        if id(x) not in seen:
            uniq.append(x)
            seen.add(id(x))
    print(f"忽略的层数: {len(uniq)}")
    return uniq


# -------------------------
# 后处理修复（照你师兄：AAttn / vanillanet / 全局 hook）
# -------------------------
def postprocess_global_mismatch(model, example_inputs):
    fixed = [0]
    hooks = []

    def make_fix_hook(name, conv: nn.Conv2d):
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if not isinstance(x, torch.Tensor):
                return
            actual_in = x.shape[1]
            is_depthwise = (conv.groups > 1 and conv.groups == conv.out_channels)

            if is_depthwise:
                expected_out = conv.weight.shape[0]
                if actual_in == expected_out:
                    return
                print(f"  [Hook] 修复 Depthwise {name}: out {expected_out} -> {actual_in}")
                with torch.no_grad():
                    w = conv.weight.data
                    if expected_out > actual_in:
                        new_w = w[:actual_in].clone()
                    else:
                        pad = actual_in - expected_out
                        pad_w = torch.zeros(pad, 1, *w.shape[2:], device=w.device, dtype=w.dtype)
                        nn.init.kaiming_uniform_(pad_w, a=math.sqrt(5))
                        new_w = torch.cat([w, pad_w], dim=0)
                    conv.weight = nn.Parameter(new_w)
                    if conv.bias is not None:
                        b = conv.bias.data
                        if expected_out > actual_in:
                            new_b = b[:actual_in].clone()
                        else:
                            pad_b = torch.zeros(pad, device=b.device, dtype=b.dtype)
                            new_b = torch.cat([b, pad_b], dim=0)
                        conv.bias = nn.Parameter(new_b)
                    conv.in_channels = actual_in
                    conv.out_channels = actual_in
                    conv.groups = actual_in
                fixed[0] += 1
            else:
                expected_in = conv.weight.shape[1]
                if actual_in == expected_in:
                    return
                print(f"  [Hook] 修复 {name}: in {expected_in} -> {actual_in}")
                with torch.no_grad():
                    w = conv.weight.data
                    if expected_in > actual_in:
                        new_w = w[:, :actual_in].clone()
                    else:
                        pad = actual_in - expected_in
                        pad_w = torch.zeros(w.shape[0], pad, *w.shape[2:], device=w.device, dtype=w.dtype)
                        nn.init.kaiming_uniform_(pad_w, a=math.sqrt(5))
                        new_w = torch.cat([w, pad_w], dim=1)
                    conv.weight = nn.Parameter(new_w)
                    conv.in_channels = actual_in
                fixed[0] += 1
        return hook

    conv_list = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    print(f"  全网 nn.Conv2d 数量: {len(conv_list)}")
    for n, c in conv_list:
        hooks.append(c.register_forward_pre_hook(make_fix_hook(n, c)))

    model.eval()
    with torch.no_grad():
        _ = model(example_inputs)

    for h in hooks:
        h.remove()
    print(f"[后处理] 全局 hook 修复完成: {fixed[0]} 处")
    return fixed[0]


def postprocess_model(model, example_inputs):
    # 这里保留“全局 hook 修复”即可（最通用）
    return postprocess_global_mismatch(model, example_inputs)


# -------------------------
# 保存为 YOLO 可加载 ckpt（dict）
# -------------------------
def args_to_dict(a):
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    if hasattr(a, "__dict__"):
        return dict(vars(a))
    return {}

def save_yolo_ckpt(model, save_path: Path, train_args: dict, extra: dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": deepcopy(model).cpu().half(),
        "ema": None,
        "optimizer": None,
        "train_args": dict(train_args),
        "yaml": str(MODEL_YAML),
        "date": time.strftime("%Y-%m-%d_%H-%M-%S"),
        **extra,
    }
    torch.save(ckpt, str(save_path))


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def scan_conv_channels(model) -> Dict[str, Dict[str, Any]]:
    """用于打印/保存：剪枝前后每个 conv/bn 的 in/out 变化（显示哪些层被剪了多少通道）"""
    info = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            info[name] = {"type": "Conv2d", "in": m.in_channels, "out": m.out_channels, "groups": m.groups}
        elif isinstance(m, nn.BatchNorm2d):
            info[name] = {"type": "BN2d", "nf": m.num_features}
    return info


def diff_channel_report(before: dict, after: dict) -> List[dict]:
    report = []
    keys = sorted(set(before.keys()) | set(after.keys()))
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        if b != a:
            report.append({"name": k, "before": b, "after": a})
    return report


def build_pruner(model, example_inputs, sparsity: float):
    imp = tp.importance.BNScaleImportance()

    customized_pruners = {
        AAttn: create_aattn_pruner(tp),
        vanillanetBlock: create_vanillanet_block_pruner(tp),
        activation: create_activation_pruner(tp),
    }
    ignored_layers = get_ignored_layers(model)

    pruner = tp.pruner.BNScalePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=sparsity,
        global_pruning=True,
        ignored_layers=ignored_layers,
        root_module_types=ROOT_MODULE_TYPES,
        customized_pruners=customized_pruners,
        reg=0.0,
    )
    return pruner, ignored_layers


def main():

    setup_torch_pruning()


    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)


    print(f"Load sparse model: {SPARSE_CKPT}")
    yolo = YOLO(SPARSE_CKPT)
    model = yolo.model

    # 确保 requires_grad（TP/DepGraph 需要）
    for p in model.parameters():
        p.requires_grad_(True)

    
    device = torch.device(DEVICE)
    model = model.to(device).eval()
    

    example_inputs = torch.randn(1, 3, DUMMY, DUMMY, device=device)

    # 剪枝前统计
    params_before = count_params(model)
    chan_before = scan_conv_channels(model)

    # DepGraph tracing
    AAttn.set_trace_mode(True)

    pruner, ignored_layers = build_pruner(model, example_inputs, sparsity=SPARSITY)

    # （可选）打印组数量
    try:
        groups = list(pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=ROOT_MODULE_TYPES))
        print(f"剪枝组数量: {len(groups)}")
    except Exception as e:
        print(f"读取剪枝组失败（可忽略）: {e}")

    print("执行物理剪枝 pruner.step() ...")
    pruner.step()
    AAttn.set_trace_mode(False)

    # 后处理（强烈建议保留）
    print("执行后处理修复 ...")
    postprocess_model(model, example_inputs)

    # 验证一次 forward
    print("验证剪枝后前向传播 ...")
    model.eval()
    with torch.no_grad():
        _ = model(example_inputs)
    print("✅ forward OK")

    # 剪枝后统计
    params_after = count_params(model)
    chan_after = scan_conv_channels(model)
    compression = (1 - params_after / max(params_before, 1)) * 100.0

    report = {
        "sparsity": SPARSITY,
        "params_before": params_before,
        "params_after": params_after,
        "compression_percent": compression,
        "channel_changes": diff_channel_report(chan_before, chan_after),
    }

    # 保存报告
    report_path = save_dir / "yolov12n-vanillanet-sparsed.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"✅ Report saved: {report_path}")

    # 保存 pruned 模型（YOLO ckpt dict，可直接 YOLO().val/predict/train）
    train_args = args_to_dict(getattr(model, "args", None))
    train_args.setdefault("task", "detect")

    pruned_path = save_dir / "yolov12n-vanillanet-sparsed.pt"
    save_yolo_ckpt(model, pruned_path, train_args, extra={
        "compression_ratio": compression,
        "original_params": params_before,
        "pruned_params": params_after,
    })
    print(f"✅ Pruned YOLO ckpt saved: {pruned_path}")

    print(f"Done. Params: {params_before:,} -> {params_after:,} ({compression:.2f}% reduced)")
    print("Tip: use pruned_yolo.pt for finetune.")


if __name__ == "__main__":
    main()
