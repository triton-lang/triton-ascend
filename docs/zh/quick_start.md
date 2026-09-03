# 快速入门

## 项目介绍

**Triton-Ascend**是适配华为Ascend处理器的Triton优化版本，用于高效进行核函数自动调优、算子编译及部署，通过兼容Triton核心语法并针对昇腾NPU特性进行深度优化，能够帮助用户在昇腾平台上快速开发和部署高性能计算任务。

本文以**Ubuntu 22.04**环境下通过软件包部署方式在线安装并运行向量加法示例为例，指导用户快速上手使用**Triton-Ascend**。如需体验更多安装方式请阅读[安装指南](./installation_guide.md)文档。

## 环境准备

**硬件要求**

支持的操作系统：linux（aarch64/x86_64）

支持的Ascend产品：Atlas A2/A3/950系列

最小硬件配置：单卡32GB内存（推荐）

**软件依赖**

确定CANN、Python和TorchNPU软件版本并安装。其中，可以参考昇腾社区官网《[CANN快速安装](https://www.hiascend.com/cann/download)》
完成驱动与固件安装。

- CANN版本：9.1.0
- Python版本：python3.11
- TorchNPU版本：2.7.1.post8

注：更多配套关系请参考[版本说明表](./release_note.md#version-compatibility-matrix)。

## 快速安装

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 快速开始

**运行tutorials中向量加法示例验证结果**

向量加法示例：[01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
通过对比Triton算子与PyTorch原生计算的输出结果，证明昇腾NPU设备可正确调用Triton算子并保证计算精度。
> ⚠️ 下述命令需在 bash 环境下执行。若使用 POSIX sh，请将 `source` 替换为 `.`。

```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取triton-ascend源码仓及用例
git clone https://github.com/triton-lang/triton-ascend.git
# 运行tutorials示例
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## 下一步：从样例到完整开发

本页帮助您完成安装与首个样例验证。若希望继续上手算子开发全流程，可按下面路径推进：

1. **算子开发**：按算子类型阅读 Vector / Cube / 融合开发教程，再对照典型样例修改 Kernel，完成开发、编译、验证闭环。
2. **自动调优**：使用 Autotune 选择合适的 Tiling 配置。
3. **调试与调优**：掌握打印调试与性能采集方法。
4. **迁移与进阶**：从 GPU Triton 迁移到 Ascend，或查阅环境变量与编译选项。

### 开发指南

| 文档 | 说明 |
|------|------|
| [Triton-Ascend 算子开发](./programming_guide/index.md) | 分核、片上内存、访存、Tiling 等通用开发原则 |
| [Triton-Ascend 算子迁移](./migration_guide/index.md) | GPU Triton 算子迁移到昇腾 NPU |
| [Triton-Ascend 算子调试与调优](./debug_guide/index.md) | 调试、性能分析与精度排查 |
| [环境变量与编译选项](./environment_variable_and_compiler_options_reference.md) | 运行与编译相关配置说明 |

### 教程与样例

| 文档 | 说明 |
|------|------|
| [Vector 算子开发](./programming_guide/vector_operator.md) | 逐元素、归约、Gather/Scatter 等 Vector Core 算子 |
| [Cube 算子开发](./programming_guide/cube_operator.md) | 以 `tl.dot`、矩阵乘为核心的 Cube 算子 |
| [融合算子开发](./programming_guide/cv_fusion_operator.md) | 同一 Kernel 中 Cube + Vector 协同的 CV 融合场景 |
| [Triton-Ascend autotune](./autotune_guide.md) | Autotune 推荐用法与自动 Tiling 适用边界 |
| [典型算子样例](./examples/index.md) | Softmax、LayerNorm、Attention、Matmul 等端到端样例 |
