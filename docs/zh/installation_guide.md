# 安装指南

## 环境准备

### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装Ascend CANN

异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

您可以访问昇腾社区官网，根据其提供的软件安装指引完成 CANN 的安装配置。

在安装过程中，CANN 版本“**{version}**”请选择如下版本之一：

**CANN版本：**

- 商用版

| Triton-Ascend版本 | CANN商用版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.3.RC1         | 2025/10/30         |

- 社区版

| Triton-Ascend版本 | CANN社区版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.5.0.alpha001  | 2025/11/12         |
|                   | CANN 8.3.RC1         | 2025/10/30         |

并根据实际环境指定CPU架构 “**{arch}**”(aarch64/x86_64)、软件版本“**{version}**”对应的软件包。

建议下载安装 8.5.0 版本:

| 软件类型    | 软件包说明       | 软件包名称                       |
|---------|------------------|----------------------------------|
| Toolkit | CANN开发套件包   | Ascend-cann-toolkit_**{version}**_linux-**{arch}**.run  |
| Ops     | CANN二进制算子包 | Ascend-cann-A3-ops_**{version}**_linux-**{arch}**.run |

注意1：A2系列的Ops包命名与A3略有区别，参考格式（ Ascend-cann-910b-ops_**{version}**_linux-**{arch}**.run ）

注意2：8.5.0之前的版本对应的Ops包的包名略有区别，参考格式（ Atlas-A3-cann-kernels_**{version}**_linux-**{arch}**.run ）

[社区下载链接](https://www.hiascend.com/developer/download/community/result?module=cann) 可以找到对应的软件包。

[社区安装指引链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit) 提供了完整的安装流程说明与依赖项配置建议，适用于需要全面部署 CANN 环境的用户。

#### CANN安装脚本

以8.5.0的A3 CANN版本为例，我们提供了脚本式安装供您参考：
```bash

# 更改run包的执行权限
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run

# 普通安装（默认安装路径：/usr/local/Ascend）
sudo ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# 默认安装路径（与 Toolkit 包一致：/usr/local/Ascend）
sudo ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
# 生效默认路径环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装CANN的python依赖
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
```

- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，${HOME}为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将```source ${HOME}/Ascend/ascend-toolkit/set_env.sh```命令写入环境变量配置文件（如.bashrc文件）。


### 安装torch_npu

当前配套的torch_npu版本为2.7.1版本。

```bash
pip install torch_npu==2.7.1
```

注：如果出现报错`ERROR: No matching distribution found for torch==2.7.1+cpu`，可以尝试手动安装torch后再安装torch_npu。
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## 通过pip安装Triton-Ascend

### 最新稳定版本
您可以通过pip安装Triton-Ascend的最新稳定版本。

```shell
pip install triton-ascend
```

- 注：如果已经安装有社区Triton，请先卸载社区Triton。再安装Triton-Ascend，避免发生冲突。
```shell
pip uninstall triton
pip install triton-ascend
```

### nightly build版本
我们为用户提供了每日更新的nightly包，用户可通过以下命令进行安装。

```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```
同时用户也能在 [历史列表](https://test.pypi.org/project/triton-ascend/#history) 中找到所有的nightly build包。

注意，如果您在执行`pip install`时遇到ssl相关报错，可追加`--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`选项解决。

## 通过源码安装Triton-Ascend

如果您需要对 Triton-Ascend 进行开发或自定义修改，则应采用源代码编译安装的方法。这种方式允许您根据项目需求调整源代码，并编译安装定制化的 Triton-Ascend 版本。

### 系统要求

- GCC >= 9.4.0
- GLIBC >= 2.27

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

### 基于LLVM构建

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。LLVM的构建支持两种构建方式，**以下两种方式二选一即可**，无需重复执行。

#### 代码准备: `git checkout` 检出指定版本的LLVM.

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

#### 方式一: clang构建安装LLVM

- 步骤1：推荐使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
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
  cd $HOME/llvm-project  # 用户git clone 拉取的 LLVM 代码路径
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

#### 方式二: GCC构建安装LLVM

- 步骤1：推荐使用clang，如果只能使用GCC安装，请注意[注1](#note1) [注2](#note2)。设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤2：执行以下命令进行构建和安装：

   ```bash
   cd $HOME/llvm-project  # your clone of LLVM.
   mkdir build
   cd build
   cmake -G Ninja  ../llvm  \
      -DLLVM_CCACHE_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm"  \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
   ninja install
   ```

<a id="note1"></a>注1：若在编译时出现错误`ld.lld: error: undefined symbol`，可在步骤2中加入设置`-DLLVM_ENABLE_LLD=ON`。

<a id="note2"></a>注2：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。

#### 克隆 Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend/python
```

#### 构建 Triton-Ascend

1. 源码安装

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

- 注3：推荐GCC >= 9.4.0，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：

- triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  取消注释后重新构建项目即可解决该问题。

2. 运行Triton示例

   安装运行时依赖，参考如下：
   ```bash
   cd triton-ascend && pip install -r requirements_dev.txt
   ```
   运行实例: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
   ```bash
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
   ```
    观察到类似的输出即说明环境配置正确。
    ```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```
