# 融合注意力（Fused Attention）

本节实现了一个基于 **Triton** 的 **Flash Attention v2 风格的融合注意力前向传播内核**，适用于昇腾（Ascend）NPU 平台。该实现支持：

- **因果（causal）与非因果注意力**
- **分块计算（tiling）以处理长序列**
- **数值稳定性优化（max-shifted softmax）**

整体结构包含两个核心 Triton 内核：

1. `_attn_fwd_inner`：执行单个 query block 与 key/value blocks 的 attention 计算（分阶段处理 causal mask）
2. `_attn_fwd`：调度所有 query blocks，并管理 block 指针、accumulator 和归一化

并通过 PyTorch `autograd.Function` 封装为可调用的 `attention` 函数，与 `torch_npu.npu_fusion_attention` 进行精度对齐验证。

完整代码见：{download}`04_fused_attention_example.py <full_examples/04_fused_attention_example.py>`

Out:

```bash
[PASSED] Attention shape:(1, 1, 128, 128), BM: 32, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(1, 1, 128, 128), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(1, 2, 256, 256), BM: 32, BN: 256, dtype: torch.bfloat16
[PASSED] Attention shape:(2, 2, 128, 256), BM: 64, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 64, 64), BM: 32, BN: 64, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 1024, 64), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(4, 32, 4096, 64), BM: 128, BN: 128, dtype: torch.float16
```

上面输出日志表明Triton和PyTorch上的输出结果完全一致。
