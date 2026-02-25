import torch
import flash_attn
# 创建一个简单的张量在GPU上运行
q = torch.randn(2, 4, 8, 64, device='cuda', dtype=torch.float16) # (batch, seq, heads, head_dim)
try:
    output = flash_attn.flash_attn_func(q, q, q)
    print("✅ FlashAttention 独立测试成功！输出形状:", output.shape)
except Exception as e:
    print("❌ FlashAttention 独立测试失败，错误详情:", e)