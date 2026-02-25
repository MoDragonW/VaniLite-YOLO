# rebuild_pruned_model.py
import torch
from ultralytics import YOLO
import sys
sys.path.insert(0, '.')

def rebuild_model():
    pruned_checkpoint_path = 'runs/detect-vanilla/pruned_depgraph/weights/pruned_model.pt'
    print(f"加载剪枝检查点: {pruned_checkpoint_path}")
    checkpoint = torch.load(pruned_checkpoint_path, map_location='cpu')
    
    # 1. 提取剪枝后的权重状态字典
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        pruned_state_dict = checkpoint['model'].state_dict()
        print(f"✓ 已提取剪枝权重，参数量: {sum(p.numel() for p in pruned_state_dict.values()):,}")
    else:
        print("✗ 检查点格式不符合预期")
        return
    
    # 2. 从正确的YAML配置创建一个全新的Detection模型
    # 注意：这个YAML必须与你训练原始基准模型时使用的配置一致！
    correct_yaml = 'yolo12-vanilanet.yaml'  # TODO: 请修改为你的正确配置文件路径
    print(f"\n从配置文件创建新模型: {correct_yaml}")
    try:
        # 这将创建一个全新的、结构正确的模型
        new_yolo = YOLO(correct_yaml)
        new_model = new_yolo.model
        print("✓ 新模型创建成功")
    except Exception as e:
        print(f"✗ 创建新模型失败: {e}")
        return
    
    # 3. 获取新模型的状态字典，并准备加载剪枝权重
    new_state_dict = new_model.state_dict()
    
    # 4. 关键步骤：将剪枝权重加载到新模型（按名称匹配）
    print("\n开始加载剪枝权重到新模型结构...")
    matched_keys = []
    mismatched_keys = []
    
    for key in new_state_dict:
        if key in pruned_state_dict:
            # 检查维度是否匹配
            if new_state_dict[key].shape == pruned_state_dict[key].shape:
                new_state_dict[key] = pruned_state_dict[key].clone()
                matched_keys.append(key)
            else:
                print(f"  形状不匹配 [跳过] {key}: 新{new_state_dict[key].shape} vs 剪枝{pruned_state_dict[key].shape}")
                mismatched_keys.append(key)
        else:
            mismatched_keys.append(key)
            # 对于新模型有而剪枝权重没有的层，保留其随机初始化（通常是Detection head的新层）
    
    # 5. 将合并后的状态字典加载回新模型
    new_model.load_state_dict(new_state_dict, strict=False)
    
    print(f"\n权重加载总结:")
    print(f"  ✓ 成功匹配并加载: {len(matched_keys)} 个参数")
    print(f"  ⚠️ 未匹配/跳过: {len(mismatched_keys)} 个参数")
    if mismatched_keys:
        print(f"    示例: {mismatched_keys[:5]}{'...' if len(mismatched_keys)>5 else ''}")
    
    # 6. 验证新模型的 nc 属性
    print("\n验证新模型关键属性:")
    if hasattr(new_model, 'nc'):
        print(f"  new_model.nc = {new_model.nc}")
    # 深度搜索 Detect 模块
    for name, module in new_model.named_modules():
        if 'Detect' in module.__class__.__name__:
            print(f"  找到 {name}: {type(module)}")
            if hasattr(module, 'nc'):
                print(f"    -> 该模块 nc = {module.nc}")
    
    # 7. 保存重建后的模型
    new_checkpoint = {
        'model': new_model,
        'compression_ratio': checkpoint.get('compression_ratio', 0),
        'original_params': checkpoint.get('original_params', 0),
        'pruned_params': checkpoint.get('pruned_params', 0),
        'note': 'Rebuilt from pruned weights with correct Detect structure'
    }
    
    output_path = 'runs/seg-vanilla/pruned_depgraph/weights/pruned_model_REBUILT.pt'
    torch.save(new_checkpoint, output_path)
    print(f"\n✅ 重建完成！模型已保存至: {output_path}")
    
    # 8. 快速前向传播验证
    print("\n运行快速推理验证...")
    try:
        new_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            output = new_model(dummy_input)
            print("✓ 前向传播成功")
            if isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if hasattr(out, 'shape'):
                        print(f"  输出{i}形状: {out.shape}")
    except Exception as e:
        print(f"✗ 前向传播验证失败: {e}")

if __name__ == '__main__':
    rebuild_model()