# 安装指南

请根据需要，选择不同的安装方式，跳转至对应步骤：

- **基于pip安装**：直接尝试使用TA的pip包选择此项。请先前往下一步<a href="#env-prepare">环境准备</a>完成前置配置，再进行pip安装操作；
- **基于源码安装**：基于TA的开发者选择此项。请先前往下一步<a href="#env-prepare">环境准备</a>完成前置配置，再选择<a href="#auto-code-base">快速安装</a>或<a href="#hand-code-base">手动安装</a>其中一种方式操作；
- **基于Docker安装**：无需环境准备，可直接跳转至<a href="#docker-build">基于Docker构建</a>进行操作

<a id="env-prepare"></a>

## 环境准备

### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装CANN

异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

您可以访问昇腾社区官网，根据其提供的[社区软件安装指引](https://www.hiascend.com/cann/download)完成 CANN 的安装与配置。开发者选择CANN版本、产品系列、CPU架构、操作系统和安装方式便可找到对应的安装命令。

在安装过程中，CANN 版本“**{version}**”请选择如下版本之一。建议下载安装 8.5.0 版本:

- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，`${HOME}`为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将```source ${HOME}/Ascend/ascend-toolkit/set_env.sh```命令写入环境变量配置文件（如.bashrc文件）。

**CANN版本：**

- 商用版

| Triton-Ascend版本 | CANN商用版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.3.RC1         | 2025/11/20<br>2025/10/30         |

- 社区版

| Triton-Ascend版本 | CANN社区版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.5.0.alpha001<br>CANN 8.3.RC1         | 2025/11/20<br>2025/11/12<br>2025/10/30         |

### 安装torch_npu

当前配套的torch_npu版本为2.7.1版本。

```bash
pip install torch_npu==2.7.1
```

注：如果出现报错`ERROR: No matching distribution found for torch==2.7.1+cpu`，可以尝试手动安装torch后再安装torch_npu。

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

<a id="pip-base"></a>

## 通过pip安装Triton-Ascend

### 最新稳定版本

您可以通过pip安装Triton-Ascend的最新稳定版本。

```shell
pip install triton-ascend
```

- 注意：社区 Triton 和 Triton-Ascend 不能同时存在。在安装依赖Triton的其他软件时，会自动安装社区 Triton，将覆盖掉已安装的 Triton-Ascend 目录。

此时也需要先卸载社区 Triton 和 Triton-Ascend，再安装 Triton-Ascend。

```shell
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

### nightly build版本

我们为用户提供了每日更新的nightly包，用户可通过以下命令进行安装。

```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```

同时用户也能在 [历史列表](https://test.pypi.org/project/triton-ascend/#history) 中找到所有的nightly build包。

注意，如果您在执行`pip install`时遇到ssl相关报错，可追加`--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`选项解决。

<a id="code-base"></a>

## 通过源码安装Triton-Ascend

如果您需要对 Triton-Ascend 进行开发或自定义修改，则应采用源代码编译安装的方法。这种方式允许您根据项目需求调整源代码，并编译安装定制化的 Triton-Ascend 版本。

在构建前，您需要完成相关构建组件的<a href="#code-require">依赖安装</a>。

我们推荐使用<a href="#auto-code-base">快速安装</a>的方式完成基于源码安装Triton-Ascend；若您有特殊需求，如目标机器无法联网等原因，可以进行<a href="#hand-code-base">手动安装</a>。

### 系统推荐

| Pytorch版本 | 推荐的GCC版本 | 推荐的GLIBC版本 |
|-------------------|----------------------|--------------------|
| PyTorch2.6.0      | (aarch64)11.2.1<br>(x86) 9.3.1 | (aarch64)>=2.28<br>(x86)>=2.17 |
| PyTorch2.7.1      | 11.2.1               | 2.28               |
| PyTorch2.8.0      | 13.3.1               | 2.28               |
| PyTorch2.9.1      | 13.3.1               | 2.28               |
| PyTorch2.10       | 13.3.1               | 2.28               |

<a id="code-require"></a>

### 依赖

#### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选择安装ccache包用于加速构建。

- 推荐版本 clang >= 15
- 推荐版本 lld >= 15

```bash
以ubuntu系统为例：
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache # optional
```

Triton-Ascend的构建强依赖zlib1g-dev，如果您使用yum源，请参考如下命令安装：

```bash
sudo yum install -y zlib-devel
```

#### 安装python依赖

```bash
pip install ninja cmake wheel pybind11 # build-time dependencies
```

<a id="auto-code-base"></a>

### 快速安装

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git
cd triton-ascend
git checkout main

# 可选，若本地有编译好的LLVM，可以直接指定本地LLVM，不会触发下载LLVM预编译包。若无，忽略这条，直接执行下面的运行安装命令即可。
export LLVM_SYSPATH=/path/to/LLVM

# 执行安装命令
pip install -e python
```

<a id="hand-code-base"></a>

### 手动安装 - 基于LLVM构建

Triton 使用 LLVM22 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。LLVM的构建支持两种构建方式，**以下两种方式二选一即可**，无需重复执行。

#### 代码准备: `git checkout` 检出指定版本的LLVM

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout fad3272286528b8a491085183434c5ad4b59ab92
   wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
   git apply fad3272.patch
   ```

#### clang构建安装LLVM

- 步骤1：使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
  如未安装，请按下面指令安装clang、lld、ccache：

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤3：执行以下命令进行构建和安装LLVM：

  ```bash
  cd {PATH_TO}/llvm_project # 路径为用户拉取LLVM代码的路径,需根据实际调整
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
  ```

- 步骤4：需要拷贝FILECHECK到目标安装路径：

   ```bash
   cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   ```

#### 克隆 Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
```

#### 构建 Triton-Ascend

- 步骤1：请确认已设置 [基于LLVM构建] 章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
- 步骤2：请确认已安装clang>=15，lld>=15，ccache

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

注1：推荐GCC版本见前段章节“系统推荐”，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：

triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  取消注释后重新构建项目即可解决该问题。

<a id="docker-build"></a>

## 基于Docker安装

我们提供了Dockerfile帮助您安装Docker环境镜像。安装过程将会自动从CANN官网中下载安装对应的CANN Toolkit和Kernel包，需要您通过`--build-arg`指定您机器需要安装的CANN相关参数。

| 参数名称 | 默认值 | 可选值                                   |
| -------- | ------ |---------------------------------------|
| CHIP_TYPE | A3     | A3、910b                               |
| CANN_VERSION | 8.5.0（推荐） | 8.5.0、8.3.RC1、8.3.RC2、8.2.RC1、8.2.RC2 |

您可以通过 npu-smi 命令查看系统上的NPU型号。

不同`CHIP_TYPE`选项对应的机器可参考：

| 选项序号 | **CHIP_TYPE 参数值** | 对应机器/产品系列 |                 典型整机                 |
| :---: |:-----------------:| :---: |:-----------------------------------:|
| 1 |       `A3`        | Atlas A3 训练系列产品 |        Atlas 900 A3 SuperPoD        |
| 2 |       `A2`        | Atlas A2 训练系列产品 |            Atlas800T A2             |

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CHIP_TYPE=A3 \
--build-arg CANN_VERSION=8.5.0 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

根据该镜像启动容器，可以参考下面的命令：

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

## 运行Triton示例

   安装运行时依赖，参考如下：

```bash
   # 拉取triton-ascend源码仓及用例（可选，非源码编译安装运行示例时需拉源码仓）
   git clone https://gitcode.com/Ascend/triton-ascend.git
   cd triton-ascend && pip install -r requirements.txt
```

   运行实例: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)

```bash
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。

```bash
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
```
