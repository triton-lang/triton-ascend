<!-- markdownlint-disable-file MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
  </picture>
</p>

<h3 align="center"><font size="68">
Triton-Ascend
</font></h3>

<p align="center">
  <a href="https://deepwiki.com/triton-lang/triton-ascend">
    <img src="https://deepwiki.com/badge.svg" alt="Ask AI on DeepWiki">
  </a>
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a href="README_zh.md"><b>中文</b></a>
</p>

<p align="center">
| <a href="https://triton-ascend.readthedocs.io/zh-cn/latest/"><b>官方文档</b></a> | <a href="https://www.hiascend.com/developer/operator?tag=triton"><b>算子开发用户旅程</b></a> | <a href="https://docs.google.com/document/d/1qfat2wZtO2nfZb5FC2dWAcR6sTqgTNSvvh7MzTDoI4s/edit?pli=1&tab=t.0"><b>社区例会</b></a> | <a href="https://www.hiascend.com/"><b>关于昇腾</b></a> |
</p>

---

## 🔥 最新消息

- [2026.04.30]Triton-Ascend 3.2.1正式版本上线
- [2026.01.20]Triton-Ascend 3.2.0正式版本上线

<div style="margin-left:1em">
<details>
<summary>更多最新消息</summary>

- [2025.11.14]Triton-Ascend3.2.0rc4预发布版本上线：<br>- [扩展tt.fp_to_fp接口，新增对FP8的类型转换支持](https://gitcode.com/Ascend/triton-ascend/pull/891) <br>- [新增scatter_ub_to_out接口，支持从UB到GM的高效数据分散操作](https://gitcode.com/Ascend/triton-ascend/pull/864)
- [2025.09.30]完善Scan/Sort类Triton Python API，支持非连续访存，完成vLLM、sglang开源仓中重点Triton算子适配
- [2025.09.19]支持Triton-Ascend [nightly包](https://test.pypi.org/project/triton-ascend/#history)提取
- [2025.08.15]完善Atomic类Triton Python API支持，完成Flaggems开源仓重点Triton算子适配，提供Matmul等简单算子高性能实现参考用例
- [2025.06.30]支持85% Triton Python API，支持连续访存，覆盖基本使用场景需求
- [2025.05.20]Triton-Ascend开源，Gitcode代码仓Alive！

</details>
</div>

---

## 📖 快速安装

### 环境准备

#### 硬件要求

支持的操作系统: linux(arch64/x86_64)

支持的Ascend产品:Atlas A2/A3/950系列

最小硬件配置: 单卡32GB显存（推荐）

#### 软件依赖

确定Python、CANN和TorchNPU软件版本并安装，软件包安装和源码编译安装均需要先完成这一步。

- Python版本选择：py3.9-py3.11 均可。

- CANN版本选择：可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成CANN的安装与配置。建议下载安装 9.0.0 版本。

- TorchNPU版本选择：当前配套的TorchNPU版本为2.7.1.post4。

### 访问昇腾NPU

如果您需要访问昇腾NPU算力资源进行开发或测试，请进入HiDevLab平台的 [HiDevLab-在线开发](https://hidevlab.huawei.com/online-develop-intro) 页面申请并使用算力。

### 快速安装

```bash
#以安装triton-ascend 3.2.1 为例
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### 源码安装

<div style="margin-left:1em">
<details>
<summary>更多源码安装</summary>

#### 安装依赖

```bash
apt updata
apt install zliblg-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

#### 编译Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
git checkout main
pip install -e .
```

#### 自定义LLVM构建（可选）

```bash
# 如果需要自定义构建LLVM过程的，可以先执行这一步再去编译Triton-Ascend
# 检出指定版本的LLVM源码并应用补丁
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout fad3272286528b8a491085183434c5ad4b59ab92
wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
git apply fad3272.patch

export LLVM_INSTALL_PREFIX=/path/to/llvm-install

# 构建自定义LLVM版本
cd {PATH_TO}/llvm_project
mkdir build
cd build
cmake ../llvm \
    -G Ninja \
    -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
ninja install

# 编译Triton-Ascend
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

</details>
</div>

### 镜像使用

<div style="margin-left:1em">
<details>
<summary>更多镜像使用</summary>

- 我们提供了Dockerfile帮助您安装Docker环境镜像。构建过程使用`quay.io/ascend/cann`预构建镜像作为基础镜像，跳过CANN安装步骤，显著加快构建速度。

- 您需要通过`--build-arg`指定`CANN_BASE_IMAGE`参数来选择适合您机器的CANN基础镜像。可用的CANN基础镜像标签可在[quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags)查看。

- 您可以通过npu-smi命令查看系统上的NPU型号。

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

- 根据该镜像启动容器，可以参考下面的命令：

```bash
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container --net=host --privileged \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
triton-ascend-image:latest \
/bin/bash

# 进入容器
docker exec -u root -it triton-ascend_container /bin/bash
```

</details>
</div>

## ✏️文档入口

- [快速开始](./docs/zh/quick_start.md)

- [在线完整文档（推荐）](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)

- [安装指南](./docs/zh/installation_guide.md)

- [架构设计与核心特性](./docs/zh/architecture_design_and_core_features.md)

- [算子开发指南](./docs/zh/programming_guide/index.md)

- [算子迁移指南](./docs/zh/migration_guide/migrate_from_gpu.md)

- [算子调试指南](./docs/zh/debug_guide/debugging.md#)

- [性能调优指南](./docs/zh/debug_guide/profiling.md#)

- [环境变量参考](./docs/zh/environment_variable_and_compiler_options_reference.md)

- [常见问题FAQ](./docs/zh/FAQ.md)

## 🏘️ 社区活动信息

1. [会议日历](https://meeting.osinfra.cn/ascend)
2. [会议纪要看板](https://docs.google.com/document/d/1qfat2wZtO2nfZb5FC2dWAcR6sTqgTNSvvh7MzTDoI4s/edit?pli=1&tab=t.0)

## 🤝 社区与贡献

- 欢迎参与Triton-Ascend的开发及代码贡献，详情请参阅 [贡献指南](./docs/zh/community/CONTRIBUTING_zh.md)

- 请通过[Issue](https://github.com/triton-lang/triton-ascend/issues)来告知我们您遇到的任何Bug。
