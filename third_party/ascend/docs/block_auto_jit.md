# JIT block 自动合并

普通 `kernel[grid](...)` 调用通过已有 Ascend 后端钩子完成候选生成、CPU 选参和结果缓存，不需要运行 `tools/` 中的脚本，也不需要提前准备 manifest。

```python
kernel[grid](...)                           # 自动选择
kernel[grid](..., block_batching_factor=1)  # 显式关闭合并
kernel[grid](..., block_batching_factor=2)  # 沿用手动 F2 的检查和回退
```

## 调用过程

1. 后端取得本次调用的 grid、dtype、shape、stride、标量和编译选项。只读取张量的主机端元信息。
2. 固定一份前端 IR，编译 F1 基线，再生成最多两个合并候选。每个候选独立编译；失败不会占用更多完整候选名额。
3. 从本次编译保存的最终 IR、LLVM、二进制和元数据检查 UB、编译路径和模型覆盖范围。
4. CV、VV 分别调用对应的 CPU 模型，选择候选并缓存决定。选参不执行 kernel，不读取历史耗时。
5. 启动时仍检查当前地址、别名、grid 等条件。检查失败时启动完整 F1，不能拼接候选与基线的元数据。

VV 保留 F2/F4 两个候选。CV 根据前端变换确认轴的含义：M/N 两轴都可合并时保留 M×4、N×4；只有一个空间方向时保留该轴 F2/F4；只有独立 batch 方向时保留 batch F2/F4。前端预检不生成设备二进制，完整候选仍不超过三个。

## 缓存与回退

选择缓存包含源码、编译选项、模型版本、工具链、grid 和参数元信息。恢复时核对基线及获选二进制的身份；地址相关的运行时检查每次重新执行。

缺失 IR、未知布局、模型不覆盖或候选被编译器拒绝时保留原因并回退 F1。F1 自身编译失败仍返回原始错误。当前数值模型沿用经过验证的 A5 配置；CV 的性能模型目前覆盖 FP32，不能将其它配置的回退解释为自动选中了性能最优。

warmup 和图捕获保留基线路径。普通 JIT 的内存缓存及磁盘缓存均可使用；可选 dynamic reuse 中的 READY tile 复用目前不登记自动合并 bundle，不能将两者的收益叠加计算。MIX 模型仍要求对应工具链与库的身份校验通过。

## 代码范围

所有新增实现、模型数据、打包适配和测试均位于 `third_party/ascend`。集成复用仓库现有的 JIT continuation 和 `compile_bundle` 钩子，未增加公共 JIT 补丁。

主要入口：`backend/block_auto_dispatch.py`；CPU 模型适配：`backend/block_auto_model.py`、`backend/block_auto_cv.py`；编译证据：`backend/block_auto_artifacts.py`。测试位于 `unittest/BlockAuto`，其中设备集成测试需显式设置 `TRITON_BLOCK_AUTO_DEVICE_TEST=1`。
