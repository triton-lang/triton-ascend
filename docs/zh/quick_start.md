# 快速入门

## 项目介绍
Triton-Ascend 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，用于高效进行核函数自动调优、算子编译及部署，通过兼容 Triton 核心语法并针对昇腾 NPU 特性进行深度优化，能够帮助用户在昇腾平台上快速开发和部署高性能计算任务。
本文以 Ubuntu 22.04 环境下通过软件包部署方式运行向量加法示例为例，指导用户快速上手使用 Triton-Ascend。

## 软件包安装

### 环境准备

#### 硬件要求

支持的操作系统: linux(aarch64/x86_64)

支持的 Ascend 产品: Atlas A2/A3/A5 系列

最小硬件配置: 单卡 32GB 内存（推荐）

#### 软件依赖

确定 Python、CANN 和 torch_npu 软件版本并安装，软件包安装和源码编译安装均需要先完成这一步。
-   Python 版本选择：py3.9-py3.11 均可。

-   CANN 版本选择：可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成 CANN 的安装与配置。建议下载安装 9.0.0 版本。

-   torch_npu 版本选择：当前配套的 torch_npu 版本为 2.7.1.post4。

### 具体实施（以whl包安装为例）
```bash
# 以安装 triton-ascend 3.2.1 为例
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```
注意：triton-ascend 3.2.1 及以上，Triton-Ascend 通过将 Triton 声明为安装依赖来缓解安装覆盖问题。 安装 Triton-Ascend 时会先安装社区 Triton，再由 Triton-Ascend 覆盖同名目录，从而避免后续安装其他依赖 Triton 的软件包时再次安装 Triton 而覆盖 Triton-Ascend。

## 快速开始

### 示例一：运行 tutorials 中向量加法示例验证结果

向量加法实例：[01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
通过对比 Triton 核函数与 PyTorch 原生计算的输出结果进行对比，证明昇腾 NPU 设备可正确调用 Triton 核函数并保证计算精度。

```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取triton-ascend源码仓及用例（可选，非源码编译安装运行示例时需拉源码仓）
git clone https://github.com/triton-lang/triton-ascend.git
# 运行tutorials示例：
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

### 示例二：从 GPU 到 NPU：迁移 Triton

Triton-Ascend 在保持与社区 Triton 语法完全兼容的同时，只需在 **张量的设备声明** 和少量 `torch.cuda.*` 接口上做替换，原有 GPU 示例即可在昇腾 NPU 上运行。
本节提供了一个简单的向量加法测试样例，采用最基础的迁移方法，对原 GPU 代码进行少量修改，演示完整的迁移过程，帮助用户快速体验 GPU 脚本迁移到昇腾 NPU 上的流程。

GPU 版本示例文件`test_add.py`如下:

```python
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
    device_id = torch.cuda.current_device()
    device = torch.device('cuda', device_id)

    x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
    output = output_cpu.cuda()

    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    torch.cuda.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```

迁移只需将 GPU 相关 API 替换为对应的 NPU 版本，对照关系如下：

| GPU 写法                         | NPU 写法                        |
| ------------------------------- | ------------------------------- |
| `device='cuda'`                 | `device='npu'`                  |
| `tensor.cuda()`                 | `tensor.npu()`                  |
| `torch.cuda.current_device()`   | `torch.npu.current_device()`    |
| `torch.cuda.synchronize()`      | `torch.npu.synchronize()`       |

`@triton.jit` 标注的核函数使用的是 Triton 通用语言一般不需要特殊修改， Launch grid 的调用方式也与 GPU 完全一致。

以 diff 形式展示核心改动：

```diff
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

# ...（kernel 代码保持不变）...

@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
-   device_id = torch.cuda.current_device()
+   device_id = torch.npu.current_device()

-   x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
-   y = torch.randn(SIZE, device='cuda', dtype=torch.float32)
+   x = torch.randn(SIZE, device='npu', dtype=torch.float32)
+   y = torch.randn(SIZE, device='npu', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
-   output = output_cpu.cuda()
+   output = output_cpu.npu()

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

-   torch.cuda.synchronize()
+   torch.npu.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```
修改完后，可用`pytest`运行用例，执行成功即表明迁移成功。
```bash
pytest test_add.py
```
若未安装`pytest`组件，可使用`pip install pytest`进行安装。
