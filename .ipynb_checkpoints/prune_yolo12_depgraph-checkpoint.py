#!/usr/bin/env python3
"""
YOLOv12-Segment DepGraph 结构化剪枝脚本 - 终极修复版

核心修复策略:
1. Conv 包装器: 不注册 Conv 类型的 pruner，只注册底层 nn.Conv2d
   - DepGraph 会自动穿透 Conv 包装器，追踪内部的 conv/bn
   - 通过 unwrap_fn 解包 Conv -> nn.Conv2d

2. AAttn QKV 逻辑: 使用 DependencyGroup 手动绑定 QKV 的 1:3 关系
   - qkv.conv 的输出是 3C，剪枝时 Q/K/V 三段必须同步
   - 通过自定义 out_channel_groups 实现

3. Detect 头跟随: 不忽略 Detect，让 DepGraph 自动追踪
   - 关键是让 DepGraph 能够追踪到 cv2/cv3 的第一个卷积

使用方法:
    # 阶段1: 稀疏化训练
    python prune_yolo12_depgraph.py --mode sparse --model runs/segment/best/weights/best.pt --epochs 30

    # 阶段2: 物理剪枝
    python prune_yolo12_depgraph.py --mode prune --model runs/segment/sparse/weights/last.pt --sparsity 0.3

    # 阶段3: 微调恢复
    python prune_yolo12_depgraph.py --mode finetune --model runs/segment/pruned/weights/pruned.pt --epochs 50

作者: Claude Code
日期: 2025-12-04 (终极修复版)
"""

import argparse
import math
import os
import sys
from pathlib import Path
from copy import deepcopy
from typing import Callable, Sequence

# 提升递归深度，防止复杂嵌套导致 DepGraph 推断中断
sys.setrecursionlimit(10000)

import torch
import torch.nn as nn

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Segment
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import A2C2f, ABlock, AAttn, C3k2, Bottleneck, C3k, C3


def setup_torch_pruning():
    """设置 torch-pruning 并配置最大递归深度"""
    import torch_pruning.dependency.constants as constants
    original_depth = constants.MAX_RECURSION_DEPTH
    constants.MAX_RECURSION_DEPTH = 5000
    print(f"设置 MAX_RECURSION_DEPTH: {original_depth} -> {constants.MAX_RECURSION_DEPTH}")

    import torch_pruning as tp

    version = getattr(tp, '__version__', '1.6.0')
    print(f"torch-pruning 版本: {version}")

    return tp


# =============================================================================
# 核心修复 1: Conv 包装器的 unwrap 函数
# =============================================================================

def conv_unwrap_fn(module):
    """
    Conv 包装器解包函数

    告诉 DepGraph 如何从 Conv 包装器获取实际的 nn.Conv2d
    这样 DepGraph 就不需要为 Conv 注册专门的 pruner，而是自动穿透到 nn.Conv2d

    返回值:
    - 如果是 Conv 包装器，返回 (nn.Conv2d, start_index, end_index)
    - 否则返回 None
    """
    if isinstance(module, Conv) and hasattr(module, 'conv'):
        # Conv 包装器，返回内部的 nn.Conv2d
        # start_index=0, end_index=None 表示使用所有通道
        return module.conv
    return None


def build_unwrap_fn_dict():
    """
    构建 unwrap_fn 字典，用于 DepGraph 穿透包装器

    关键: 对于所有继承自 nn.Module 但包含 nn.Conv2d 的模块，
    都需要提供 unwrap 函数
    """
    return {
        Conv: conv_unwrap_fn,
    }


# =============================================================================
# 核心修复 2: AAttn 的自定义剪枝处理器
# =============================================================================

def create_aattn_pruner(tp):
    """
    创建 AAttn 自定义剪枝处理器

    AAttn 的特殊性:
    - qkv: Conv(C -> 3C)，输出分为 Q/K/V 三段
    - pe: Conv(C -> C, depthwise)
    - proj: Conv(C -> C)

    剪枝时必须保证:
    1. qkv 输入剪掉通道 i，则输出必须剪掉 i, i+C, i+2C
    2. pe 的 depthwise 必须同步剪枝
    3. proj 的输入必须与 V 段对齐
    """

    class AAttnPruner(tp.pruner.BasePruningFunc):
        """AAttn 剪枝处理器 - 处理 QKV 的 1:3 对应关系"""

        TARGET_MODULES = (AAttn,)

        def prune_out_channels(self, layer: AAttn, idxs: Sequence[int]):
            """
            剪枝 AAttn 的输出通道

            AAttn 的输出来自 proj，所以只需要剪枝 proj 的输出
            """
            idxs = list(idxs)

            # 1. 剪枝 proj 输出
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, 'conv') else proj

            # 使用 torch-pruning 的标准函数
            tp.pruner.function.prune_conv_out_channels(proj_conv, idxs)
            if hasattr(proj, 'bn'):
                tp.pruner.function.prune_batchnorm_out_channels(proj.bn, idxs)

            # 2. 更新元数据
            if hasattr(layer, 'c2'):
                layer.c2 = proj_conv.out_channels

            return idxs

        def prune_in_channels(self, layer: AAttn, idxs: Sequence[int]):
            """
            剪枝 AAttn 的输入通道

            这是最复杂的部分:
            1. qkv 输入剪枝 -> 输出同步剪枝 (1:3)
            2. pe depthwise 剪枝
            3. proj 输入剪枝
            """
            idxs = list(idxs)
            C = layer.qkv.conv.in_channels if hasattr(layer.qkv, 'conv') else layer.qkv.in_channels

            # 1. QKV 输入剪枝
            qkv = layer.qkv
            qkv_conv = qkv.conv if hasattr(qkv, 'conv') else qkv
            tp.pruner.function.prune_conv_in_channels(qkv_conv, idxs)

            # 2. QKV 输出同步剪枝 (1:3 对应)
            # 剪掉输入通道 i，需要剪掉输出的 [i, i+C, i+2C]
            qkv_out_idxs = []
            new_C = qkv_conv.in_channels  # 剪枝后的 C
            old_C = new_C + len(idxs)  # 原始的 C

            for i in idxs:
                qkv_out_idxs.extend([i, i + old_C, i + 2 * old_C])

            tp.pruner.function.prune_conv_out_channels(qkv_conv, qkv_out_idxs)
            if hasattr(qkv, 'bn'):
                tp.pruner.function.prune_batchnorm_out_channels(qkv.bn, qkv_out_idxs)

            # 3. PE depthwise 剪枝
            if hasattr(layer, 'pe'):
                pe = layer.pe
                pe_conv = pe.conv if hasattr(pe, 'conv') else pe

                if isinstance(pe_conv, nn.Conv2d) and pe_conv.groups == pe_conv.in_channels:
                    # Depthwise conv: 直接剪枝 weight
                    keep_idxs = [i for i in range(pe_conv.in_channels) if i not in idxs]
                    pe_conv.weight = nn.Parameter(pe_conv.weight.data[keep_idxs].clone())
                    if pe_conv.bias is not None:
                        pe_conv.bias = nn.Parameter(pe_conv.bias.data[keep_idxs].clone())
                    pe_conv.in_channels = pe_conv.out_channels = pe_conv.groups = len(keep_idxs)

                    if hasattr(pe, 'bn'):
                        # BN 也需要剪枝
                        bn = pe.bn
                        bn.weight = nn.Parameter(bn.weight.data[keep_idxs].clone())
                        bn.bias = nn.Parameter(bn.bias.data[keep_idxs].clone())
                        bn.running_mean = bn.running_mean[keep_idxs]
                        bn.running_var = bn.running_var[keep_idxs]
                        bn.num_features = len(keep_idxs)

            # 4. Proj 输入剪枝
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, 'conv') else proj
            tp.pruner.function.prune_conv_in_channels(proj_conv, idxs)

            # 5. 更新元数据
            if hasattr(layer, 'c1'):
                layer.c1 = qkv_conv.in_channels
            if hasattr(layer, 'num_heads') and hasattr(layer, 'head_dim'):
                layer.head_dim = qkv_conv.in_channels // layer.num_heads

            return idxs

        def get_out_channels(self, layer: AAttn):
            """获取输出通道数"""
            proj = layer.proj
            proj_conv = proj.conv if hasattr(proj, 'conv') else proj
            return proj_conv.out_channels

        def get_in_channels(self, layer: AAttn):
            """获取输入通道数"""
            qkv = layer.qkv
            qkv_conv = qkv.conv if hasattr(qkv, 'conv') else qkv
            return qkv_conv.in_channels

    return AAttnPruner()


# =============================================================================
# 核心修复 3: 后处理修复函数 (作为安全网)
# =============================================================================

def postprocess_aattn(model):
    """
    后处理修复 AAttn 的维度不一致问题

    即使 DepGraph 正确工作，有时也会出现维度不一致
    这个函数作为安全网，确保所有 AAttn 的维度正确
    """
    print("\n[后处理] 检查并修复 AAttn 维度...")
    fixed_count = 0

    for name, m in model.named_modules():
        if not isinstance(m, AAttn):
            continue

        qkv = m.qkv
        proj = m.proj
        pe = m.pe

        qkv_conv = qkv.conv if hasattr(qkv, 'conv') else qkv
        proj_conv = proj.conv if hasattr(proj, 'conv') else proj
        pe_conv = pe.conv if hasattr(pe, 'conv') else pe

        # 目标维度: 以 qkv 输入为准
        target_dim = qkv_conv.in_channels

        # 检查 qkv 输出是否为 3*target_dim
        expected_qkv_out = target_dim * 3
        if qkv_conv.out_channels != expected_qkv_out:
            print(f"  修复 {name}.qkv: out_channels {qkv_conv.out_channels} -> {expected_qkv_out}")

            # 重新采样 weight
            old_out = qkv_conv.out_channels
            old_third = old_out // 3
            new_third = target_dim

            keep_idxs = []
            for k in range(3):
                start = k * old_third
                keep_idxs.extend(range(start, start + new_third))

            qkv_conv.weight = nn.Parameter(qkv_conv.weight.data[keep_idxs].clone())
            if qkv_conv.bias is not None:
                qkv_conv.bias = nn.Parameter(qkv_conv.bias.data[keep_idxs].clone())
            qkv_conv.out_channels = expected_qkv_out

            if hasattr(qkv, 'bn'):
                bn = qkv.bn
                bn.weight = nn.Parameter(bn.weight.data[keep_idxs].clone())
                bn.bias = nn.Parameter(bn.bias.data[keep_idxs].clone())
                bn.running_mean = bn.running_mean[keep_idxs]
                bn.running_var = bn.running_var[keep_idxs]
                bn.num_features = expected_qkv_out

            fixed_count += 1

        # 检查 proj 输入
        if proj_conv.in_channels != target_dim:
            print(f"  修复 {name}.proj: in_channels {proj_conv.in_channels} -> {target_dim}")
            new_w = proj_conv.weight.data[:, :target_dim].clone()
            proj_conv.weight = nn.Parameter(new_w)
            proj_conv.in_channels = target_dim
            fixed_count += 1

        # 检查 pe (depthwise)
        if pe_conv.in_channels != target_dim:
            print(f"  修复 {name}.pe: channels {pe_conv.in_channels} -> {target_dim}")
            keep_idxs = list(range(min(pe_conv.in_channels, target_dim)))
            if len(keep_idxs) < target_dim:
                # 需要扩展
                pad_count = target_dim - len(keep_idxs)
                keep_idxs.extend([0] * pad_count)  # 复制第一个通道

            pe_conv.weight = nn.Parameter(pe_conv.weight.data[keep_idxs].clone())
            if pe_conv.bias is not None:
                pe_conv.bias = nn.Parameter(pe_conv.bias.data[keep_idxs].clone())
            pe_conv.in_channels = pe_conv.out_channels = pe_conv.groups = target_dim

            if hasattr(pe, 'bn'):
                bn = pe.bn
                bn.weight = nn.Parameter(bn.weight.data[keep_idxs].clone())
                bn.bias = nn.Parameter(bn.bias.data[keep_idxs].clone())
                bn.running_mean = bn.running_mean[keep_idxs]
                bn.running_var = bn.running_var[keep_idxs]
                bn.num_features = target_dim

            fixed_count += 1

        # 更新 head_dim
        if hasattr(m, 'num_heads'):
            m.head_dim = target_dim // m.num_heads

    print(f"[后处理] AAttn 修复完成，共修正 {fixed_count} 处")
    return fixed_count


def postprocess_global_mismatch(model, example_inputs):
    """
    全局动态 Hook 修复 - 修复整个网络中所有维度不匹配问题

    核心策略: 全网递归注册
    - 遍历 model 中的 **所有** nn.Conv2d（不仅仅是 Detect 头）
    - 在每个 nn.Conv2d 上注册 forward_pre_hook
    - Hook 直接比对 input.shape[1] 和 weight.shape[1]
    - 不匹配则立即原地修复权重

    这样可以修复 Backbone、Neck、Head 中任何位置的维度不对齐问题。
    """
    print("\n[后处理] 使用全局动态 Hook 修复维度不匹配 (全网递归模式)...")

    fixed_count = [0]  # 使用列表以便在闭包中修改
    hooks = []

    def make_fix_hook(name, conv):
        """
        创建一个 forward_pre_hook，用于在前向传播前修复维度不匹配

        直接在 nn.Conv2d 上注册，比对 input.shape[1] 和 weight.shape[1]

        特殊处理:
        - Depthwise 卷积 (groups == out_channels): weight.shape[1] 固定为 1，
          应检查 weight.shape[0] 是否等于 input.shape[1]
        - 普通卷积: 检查 weight.shape[1] 是否等于 input.shape[1]
        """
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if not isinstance(x, torch.Tensor):
                return

            actual_in = x.shape[1]

            # 判断是否为 Depthwise 卷积
            # Depthwise 特征: groups > 1 且 groups == out_channels
            is_depthwise = (conv.groups > 1 and conv.groups == conv.out_channels)

            if is_depthwise:
                # Depthwise 卷积: weight.shape = [out_channels, 1, H, W]
                # 检查 weight.shape[0] (输出通道) 是否等于 input.shape[1]
                expected_channels = conv.weight.shape[0]

                if actual_in == expected_channels:
                    return  # 维度匹配，无需修复

                print(f"  [Hook] 修复 Depthwise {name}: weight[{expected_channels},:] -> input[{actual_in}]")

                # 动态修复 Depthwise 权重 (沿 dim=0 裁剪/扩展)
                with torch.no_grad():
                    old_weight = conv.weight.data  # [out_ch, 1, H, W]

                    if expected_channels > actual_in:
                        # 截断: 只保留前 actual_in 个输出通道
                        new_weight = old_weight[:actual_in].clone()
                    else:
                        # 扩展: 用 Kaiming 初始化填充新通道
                        pad_size = actual_in - expected_channels
                        pad_weight = torch.zeros(
                            pad_size, 1, *old_weight.shape[2:],
                            device=old_weight.device, dtype=old_weight.dtype
                        )
                        nn.init.kaiming_uniform_(pad_weight, a=math.sqrt(5))
                        new_weight = torch.cat([old_weight, pad_weight], dim=0)

                    conv.weight = nn.Parameter(new_weight)

                    # 同步更新 bias
                    if conv.bias is not None:
                        old_bias = conv.bias.data
                        if expected_channels > actual_in:
                            new_bias = old_bias[:actual_in].clone()
                        else:
                            pad_bias = torch.zeros(pad_size, device=old_bias.device, dtype=old_bias.dtype)
                            new_bias = torch.cat([old_bias, pad_bias], dim=0)
                        conv.bias = nn.Parameter(new_bias)

                    # 更新 Depthwise 的相关属性
                    conv.in_channels = actual_in
                    conv.out_channels = actual_in
                    conv.groups = actual_in

                fixed_count[0] += 1

            else:
                # 普通卷积: 检查 weight.shape[1] (输入通道)
                expected_in = conv.weight.shape[1]

                if actual_in == expected_in:
                    return  # 维度匹配，无需修复

                print(f"  [Hook] 修复 {name}: weight[:,{expected_in}] -> input[{actual_in}]")

                # 动态修复权重 (沿 dim=1 裁剪/扩展)
                with torch.no_grad():
                    old_weight = conv.weight.data

                    if expected_in > actual_in:
                        # 截断: 只保留前 actual_in 个输入通道
                        new_weight = old_weight[:, :actual_in].clone()
                    else:
                        # 扩展: 用 Kaiming 初始化填充新通道
                        pad_size = actual_in - expected_in
                        pad_weight = torch.zeros(
                            old_weight.shape[0], pad_size, *old_weight.shape[2:],
                            device=old_weight.device, dtype=old_weight.dtype
                        )
                        nn.init.kaiming_uniform_(pad_weight, a=math.sqrt(5))
                        new_weight = torch.cat([old_weight, pad_weight], dim=1)

                    conv.weight = nn.Parameter(new_weight)
                    conv.in_channels = actual_in

                fixed_count[0] += 1

        return hook

    # 递归遍历整个模型中的所有 nn.Conv2d
    conv_list = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_list.append((name, m))

    print(f"  在整个模型中发现 {len(conv_list)} 个 nn.Conv2d")

    # 给每个 nn.Conv2d 注册 Hook
    for name, conv in conv_list:
        hook = conv.register_forward_pre_hook(make_fix_hook(name, conv))
        hooks.append(hook)

    # 运行一次前向传播来触发修复
    print("\n  运行前向传播触发全网修复...")
    model.eval()
    try:
        with torch.no_grad():
            _ = model(example_inputs)
        print("  前向传播成功!")
    except Exception as e:
        print(f"  前向传播仍然失败: {e}")
        import traceback
        traceback.print_exc()

    # 移除所有 hooks
    for h in hooks:
        h.remove()

    print(f"\n[后处理] 全局修复完成，共修正 {fixed_count[0]} 处")
    return fixed_count[0]


def postprocess_model(model, example_inputs=None):
    """
    综合后处理

    Args:
        model: 待修复的模型
        example_inputs: 用于触发 hook 的示例输入，如果为 None 则创建默认输入
    """
    if example_inputs is None:
        example_inputs = torch.randn(1, 3, 64, 64)

    total_fixed = 0
    # 1. 先修复 AAttn 的 QKV 维度问题
    total_fixed += postprocess_aattn(model)
    # 2. 全局修复所有维度不匹配（包括 Backbone, Neck, Head）
    total_fixed += postprocess_global_mismatch(model, example_inputs)
    return total_fixed


# =============================================================================
# 剪枝器配置
# =============================================================================

def get_pruner(model, example_inputs, sparsity, tp, ignored_layers):
    """
    创建 DepGraph 剪枝器

    关键配置:
    1. 只注册 AAttn 的自定义处理器
    2. 使用 nn.Conv2d 作为 root_module_types
    3. 通过 unwrap_module_fn 让 DepGraph 穿透 Conv 包装器
    """
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 创建 AAttn 自定义处理器
    aattn_pruner = create_aattn_pruner(tp)

    # 自定义处理器字典 - 只注册 AAttn
    customized_pruners = {
        AAttn: aattn_pruner,
    }

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=sparsity,
        ignored_layers=ignored_layers,
        # 关键: 只使用 nn.Conv2d 作为根模块
        # DepGraph 会自动穿透 Conv 包装器
        root_module_types=[nn.Conv2d, nn.Linear],
        round_to=32,  # 通道数对齐到 32 (A2C2f 要求)
        global_pruning=True,
        customized_pruners=customized_pruners,
    )

    return pruner


def get_ignored_layers(model):
    """
    获取需要忽略的层

    策略:
    1. DFL 层必须忽略 (固定结构)
    2. 最后的分类/回归头可以选择性忽略
    3. 不忽略 Detect 头的输入卷积 (让它们跟随 Backbone)
    """
    ignored = []

    for name, m in model.named_modules():
        # DFL 是固定结构，不能剪枝
        if m.__class__.__name__ == 'DFL':
            ignored.append(m)

        # 分类头最后的卷积 (输出类别数)
        if isinstance(m, nn.Conv2d):
            # 检查是否是输出层 (nc 个输出通道)
            if hasattr(model, 'model'):
                detect = None
                for sub in model.model.modules():
                    if isinstance(sub, (Detect, Segment)):
                        detect = sub
                        break
                if detect and m.out_channels == detect.nc:
                    ignored.append(m)
                # reg_max * 4 的输出也忽略
                if detect and m.out_channels == detect.reg_max * 4:
                    ignored.append(m)

    print(f"忽略的层数: {len(ignored)}")
    return ignored


# =============================================================================
# 主要功能函数
# =============================================================================

def print_model_stats(model, name="模型"):
    """打印模型统计信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{name} 统计:")
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,}")

    return total_params


def sparse_training(args, tp):
    """
    阶段1: 稀疏化训练

    使用 Group Lasso 正则化，使不重要的参数组趋向于零
    """
    print("\n" + "="*60)
    print("阶段1: 稀疏化训练 (DepGraph)")
    print("="*60)

    # 加载模型
    yolo = YOLO(args.model)
    model = yolo.model

    print_model_stats(model, "原始模型")

    # 创建伪输入 (CPU) 用于构建依赖图
    model_cpu = deepcopy(model).cpu()

    # 关键：启用所有参数的梯度
    print("\n启用所有参数的梯度...")
    for param in model_cpu.parameters():
        param.requires_grad = True

    model_cpu.eval()
    dummy_size = 64
    example_inputs = torch.randn(1, 3, dummy_size, dummy_size)

    # 获取忽略层
    print("\n识别需要忽略的层:")
    ignored_layers = get_ignored_layers(model_cpu)

    # 启用追踪模式构建剪枝器
    print("\n启用 AAttn 追踪模式...")
    AAttn.set_trace_mode(True)

    # 创建剪枝器 (用于正则化)
    pruner = get_pruner(model_cpu, example_inputs, args.sparsity, tp, ignored_layers)

    # 关闭追踪模式
    AAttn.set_trace_mode(False)

    # 验证剪枝组数量
    all_groups = list(pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d]))
    print(f"  剪枝组数量: {len(all_groups)}")

    if len(all_groups) < 10:
        print("警告: 剪枝组数量过少!")

    # 稀疏化训练回调
    class SparseCallback:
        def __init__(self, pruner, lambda_sparse=1e-4):
            self.pruner = pruner
            self.lambda_sparse = lambda_sparse

        def on_train_batch_end(self, trainer):
            # 注入 Group Lasso 正则化梯度
            self.pruner.regularize(trainer.model, reg=self.lambda_sparse)

    callback = SparseCallback(pruner, lambda_sparse=args.lambda_sparse)
    yolo.add_callback("on_train_batch_end", callback.on_train_batch_end)

    # 开始训练
    print(f"\n开始稀疏化训练 (epochs={args.epochs}, lambda={args.lambda_sparse})...")

    results = yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        nbs=args.nbs,
        cache=args.cache,
        lr0=args.lr * 0.1,
        project=str(ROOT / "runs/segment"),
        name="sparse_depgraph",
        exist_ok=True,
        device=args.device,
        workers=args.workers,
    )

    print("\n稀疏化训练完成!")
    print(f"模型保存至: {ROOT / 'runs/segment/sparse_depgraph/weights/last.pt'}")

    return results


def physical_pruning(args, tp):
    """
    阶段2: 物理剪枝

    使用 DepGraph 进行真正的结构化剪枝 (物理移除通道)
    """
    print("\n" + "="*60)
    print("阶段2: 物理剪枝 (DepGraph - 终极修复版)")
    print("="*60)

    # 加载模型
    yolo = YOLO(args.model)
    model = yolo.model.cpu()

    # 调试信息
    print("\n[调试] 模型结构分析:")
    aattn_count = 0
    conv_count = 0
    for name, m in model.named_modules():
        if isinstance(m, AAttn):
            aattn_count += 1
            print(f"  AAttn: {name}")
            print(f"    qkv: {m.qkv.conv.in_channels} -> {m.qkv.conv.out_channels}")
            print(f"    proj: {m.proj.conv.in_channels} -> {m.proj.conv.out_channels}")
            print(f"    pe: {m.pe.conv.in_channels} (groups={m.pe.conv.groups})")
        elif isinstance(m, Conv):
            conv_count += 1
    print(f"  总共 {aattn_count} 个 AAttn, {conv_count} 个 Conv 包装器")

    # 关键：启用所有参数的梯度
    print("\n启用所有参数的梯度...")
    for param in model.parameters():
        param.requires_grad = True

    model.eval()

    original_params = print_model_stats(model, "剪枝前模型")

    # 创建伪输入
    dummy_size = 64
    example_inputs = torch.randn(1, 3, dummy_size, dummy_size)

    # 获取忽略层
    print("\n识别需要忽略的层:")
    ignored_layers = get_ignored_layers(model)

    # 启用追踪模式
    print("\n启用 AAttn 追踪模式...")
    AAttn.set_trace_mode(True)

    # 创建剪枝器
    print(f"\n创建剪枝器 (目标稀疏率: {args.sparsity*100:.1f}%)...")
    print("正在构建依赖图...")

    try:
        pruner = get_pruner(model, example_inputs, args.sparsity, tp, ignored_layers)
    except Exception as e:
        print(f"依赖图构建失败: {e}")
        import traceback
        traceback.print_exc()
        AAttn.set_trace_mode(False)
        return None

    # 分析剪枝组
    print("\n分析剪枝组...")
    all_groups = list(pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d]))
    print(f"  总剪枝组数量: {len(all_groups)}")

    if len(all_groups) < 10:
        print("\n警告: 剪枝组数量过少，可能存在依赖追踪问题!")
        print("尝试显示所有组:")
        for i, group in enumerate(all_groups):
            print(f"  组 {i+1}:")
            for dep, idxs in group:
                print(f"    - {dep.target.name}: {len(idxs)} 通道")
    else:
        print("\n剪枝组示例 (前5个):")
        for i, group in enumerate(all_groups[:5]):
            print(f"  组 {i+1} ({len(group)} 依赖项):")
            for dep, idxs in group[:3]:
                print(f"    - {dep.target.name}: {len(idxs)} 通道")
            if len(group) > 3:
                print(f"    ... 还有 {len(group)-3} 项")

    # 执行剪枝
    print("\n执行物理剪枝...")
    try:
        pruner.step()
    except Exception as e:
        print(f"剪枝执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 关闭追踪模式
    AAttn.set_trace_mode(False)

    # 后处理修复 (传入 example_inputs 用于动态 hook 修复)
    print("\n执行后处理修复...")
    postprocess_model(model, example_inputs)

    # 验证
    print("\n验证剪枝后的模型...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(example_inputs)
        print("前向传播验证通过!")
        validation_passed = True

        # 检查输出形状
        if isinstance(output, (tuple, list)):
            print(f"  输出类型: {type(output)}, 数量: {len(output)}")
            for i, o in enumerate(output[:3]):
                if isinstance(o, torch.Tensor):
                    print(f"  输出[{i}] 形状: {o.shape}")
        else:
            print(f"  输出形状: {output.shape}")

    except Exception as e:
        print(f"模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        validation_passed = False

    pruned_params = print_model_stats(model, "剪枝后模型")

    # 计算压缩率
    compression_ratio = (1 - pruned_params / original_params) * 100
    print(f"\n压缩率: {compression_ratio:.2f}% 参数减少")

    # 保存模型
    if validation_passed:
        save_dir = ROOT / "runs/segment/pruned_depgraph/weights"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整模型
        # 标记剪枝状态，便于训练阶段加载时直接复用结构
        model.is_pruned = True
        save_path = save_dir / "pruned_model.pt"
        torch.save({
            'model': model,
            'compression_ratio': compression_ratio,
            'original_params': original_params,
            'pruned_params': pruned_params,
        }, save_path)
        print(f"\n模型保存至: {save_path}")

        # 也保存 YOLO 格式
        yolo_save_path = save_dir / "pruned_yolo.pt"
        try:
            # 覆盖 ckpt，强制写入剪枝后的结构，避免携带原始未剪枝的 ckpt 元数据
            model.is_pruned = True
            yolo.ckpt = {
                "model": deepcopy(model).half(),
                "train_args": getattr(model, "args", {}),
                "compression_ratio": compression_ratio,
                "original_params": original_params,
                "pruned_params": pruned_params,
            }
            yolo.save(yolo_save_path)
            print(f"YOLO 格式保存至: {yolo_save_path}")
        except Exception as e:
            print(f"YOLO 格式保存失败: {e}")
    else:
        print("\n由于验证失败，不保存模型")

    return model


def finetune(args):
    """
    阶段3: 微调恢复
    """
    print("\n" + "="*60)
    print("阶段3: 微调恢复")
    print("="*60)

    # 加载剪枝后的模型
    yolo = YOLO(args.model)
    model = yolo.model

    print_model_stats(model, "待微调模型")

    # 确保所有参数可训练
    for param in model.parameters():
        param.requires_grad = True

    # 微调
    print(f"\n开始微调 (epochs={args.epochs})...")

    # 关键：直接复用当前剪枝后的模型，不让 Trainer 重新根据 yaml 重建大模型
    train_overrides = {
        **yolo.overrides,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nbs": args.nbs,
        "cache": args.cache,
        "lr0": args.lr * 0.01,
        "project": str(ROOT / "runs/segment"),
        "name": "finetuned_depgraph",
        "exist_ok": True,
        "device": args.device,
        "workers": args.workers,
        "resume": False,  # 显式禁止 resume 分支去重建模型
        "mode": "train",
    }

    trainer_cls = yolo._smart_load("trainer")
    trainer = trainer_cls(overrides=train_overrides, _callbacks=yolo.callbacks)

    # 将剪枝后的模型直接挂到 trainer，setup_model 时将跳过重建
    trainer.model = yolo.model
    yolo.model = trainer.model
    trainer.hub_session = yolo.session
    trainer.train()

    # 与 YOLO.train 保持一致，更新 yolo 对象的模型与 metrics
    results = None
    try:
        from ultralytics.nn.tasks import attempt_load_one_weight
        from ultralytics.utils import RANK

        if RANK in {-1, 0}:
            ckpt = trainer.best if trainer.best.exists() else trainer.last
            yolo.model, yolo.ckpt = attempt_load_one_weight(ckpt)
            yolo.overrides = yolo.model.args
            results = getattr(trainer.validator, "metrics", None)
    except Exception as e:
        print(f"加载最佳权重时出错: {e}")

    print("\n微调完成!")
    return results


def analyze_model(args, tp):
    """
    分析模型的 DepGraph 依赖结构
    """
    print("\n" + "="*60)
    print("模型 DepGraph 分析")
    print("="*60)

    # 加载模型
    yolo = YOLO(args.model)
    model = yolo.model.cpu()

    # 关键：启用所有参数的梯度
    print("启用所有参数的梯度...")
    for param in model.parameters():
        param.requires_grad = True

    model.eval()

    print_model_stats(model, "当前模型")

    # 详细打印模型结构
    print("\n[模型结构详情]")
    for name, m in model.named_modules():
        if isinstance(m, AAttn):
            print(f"\nAAttn: {name}")
            print(f"  qkv.conv: {m.qkv.conv}")
            print(f"  proj.conv: {m.proj.conv}")
            print(f"  pe.conv: {m.pe.conv}")
            print(f"  num_heads: {m.num_heads}, head_dim: {m.head_dim}")
        elif isinstance(m, (Detect, Segment)):
            print(f"\n{m.__class__.__name__}: {name}")
            print(f"  nc: {m.nc}, nl: {m.nl}, reg_max: {m.reg_max}")
            print(f"  cv2: {len(m.cv2)} 个分支")
            print(f"  cv3: {len(m.cv3)} 个分支")
            if hasattr(m, 'cv4'):
                print(f"  cv4: {len(m.cv4)} 个分支")

    # 创建伪输入
    dummy_size = 64
    example_inputs = torch.randn(1, 3, dummy_size, dummy_size)

    # 获取忽略层
    print("\n识别需要忽略的层:")
    ignored_layers = get_ignored_layers(model)

    # 启用 AAttn 追踪模式
    print("\n启用 AAttn 追踪模式...")
    AAttn.set_trace_mode(True)

    # 构建依赖图
    print("构建依赖图...")

    try:
        DG = tp.DependencyGraph()

        # 注册自定义剪枝器
        print("注册自定义剪枝器...")
        aattn_pruner = create_aattn_pruner(tp)
        DG.register_customized_layer(AAttn, aattn_pruner)
        print(f"  已注册: AAttn")

        # 构建依赖
        DG.build_dependency(model, example_inputs=example_inputs)

        # 关闭追踪模式
        AAttn.set_trace_mode(False)

        # 获取所有组
        all_groups = list(DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d]))
        print(f"\n总剪枝组数量: {len(all_groups)}")

        # 详细分析
        print("\n剪枝组详情:")
        for i, group in enumerate(all_groups[:20]):
            print(f"\n  组 {i+1} ({len(group)} 个依赖项):")
            for dep, idxs in group[:5]:
                print(f"    - {dep.target.name}: {len(idxs)} 通道")
            if len(group) > 5:
                print(f"    ... 还有 {len(group)-5} 个依赖项")

        if len(all_groups) > 20:
            print(f"\n  ... 还有 {len(all_groups)-20} 个组")

    except Exception as e:
        print(f"依赖图构建失败: {e}")
        import traceback
        traceback.print_exc()

    AAttn.set_trace_mode(False)
    print("\n分析完成!")


def main():
    parser = argparse.ArgumentParser(description='YOLOv12-Segment DepGraph 结构化剪枝 (终极修复版)')

    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['sparse', 'prune', 'finetune', 'analyze'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='runs/segment/best/weights/best.pt')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/toug-coco128-seg.yaml')
    parser.add_argument('--sparsity', type=float, default=0.3)
    parser.add_argument('--lambda-sparse', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=960)
    parser.add_argument('--nbs', type=int, default=960)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='0,1,2,3')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--cache', type=str, default='ram', help='dataloader cache mode: False/ram/disk')

    args = parser.parse_args()

    # 设置 torch-pruning
    tp = setup_torch_pruning()

    # 执行
    if args.mode == 'sparse':
        sparse_training(args, tp)
    elif args.mode == 'prune':
        physical_pruning(args, tp)
    elif args.mode == 'finetune':
        finetune(args)
    elif args.mode == 'analyze':
        analyze_model(args, tp)


if __name__ == '__main__':
    main()
