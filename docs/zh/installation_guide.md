# 快速安装

本文主要介绍在 Ubuntu 环境中如何快速完成 **Triton-Ascend** 基础配套的安装，如果需要详细的操作步骤请参考[<u>《安装介绍》</u>]( #anzhuangzhinan )。

## 基于Docker镜像快速搭建
直接使用 Triton-Ascend 发布的开箱即用的镜像，快速构筑开发环境。

### 确认镜像
**表1** 昇腾芯片与对应产品及镜像 tag 部分对照表。更多镜像参见 OVERVIEW.zh.md 文档。
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">芯片型号</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">对应产品</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">镜像tag</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">昇腾910b</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A2、Atlas 900 A2 PoD</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">昇腾A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">昇腾950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 950PR系列</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-950-ubuntu22.04-py3.11</td>
  </tr>
</table>

### 具体实施
1.  创建容器

    ```
    # 假设您的NPU设备型号是A3,且设备安装在/dev/davinci1上，并且您的NPU驱动程序安装在/usr/local/Ascend上：
    # 以image_tag：3.2.1-a3-ubuntu22.04-py3.11为例：
    container_name=triton-ascend_container
    image_tag=3.2.1-a3-ubuntu22.04-py3.11
    docker run -u 0 -dit --shm-size=512g --name=${container_name} --net=host --privileged \
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
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home:/home \
    quay.io/ascend/triton:${image_tag} \
    /bin/bash
    ```

2.  进入容器
    ```
    docker exec -it triton-ascend_container bash
    ```

3.  运行实例：<a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>
    观察到类似的输出即说明环境已搭建完成。
    ```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```

# 安装介绍 <a id = "anzhuangzhinan" ></a>

## 概述

Triton-Ascend 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，主要用于提供高效的核函数自动调优、算子编译及部署能力，支持 Ascend Atlas A2/A3 等系列产品， 兼容 Triton 核心语法的同时，针对昇腾 NPU 特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

本文主要介绍 Triton-Ascend 的三种安装方式：软件包安装；镜像包安装；源码编译安装。

## 硬件和操作系统

-   Ascend 产品：支持 Atlas A2/A3 系列。

-   NPU 配置：建议至少单卡 32GB 显存。

-   操作系统：需 Linux 系统，具体选择请参考<a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">兼容性查询助手</a>。本文接下来所有操作均以 Ubuntu 环境演示。


## 安装方式选择

快速决策，绝大多数用户直接选择基于软件包安装即可。需要容器化部署选镜像包安装，需要二次开发改代码选源码编译安装。
**表2** 各安装方式对比差异表
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 24.41%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">安装方式</th>
    <th style="width: 19.15%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">适用人群</th>
    <th style="width: 26.21%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">核心优势</th>
    <th style="width: 30.23%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">选择理由</th>
  </tr>
    <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">软件包安装</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">生产环境用户、运维人员</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">安装简单、依赖自动管理、升级卸载方便</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">追求稳定、快速部署，不想折腾环境</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">源码编译安装</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">开发者、需要自定义功能的用户</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">可定制化程度高、支持最新功能</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">需要修改源码、适配特殊硬件或功能</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">镜像包安装</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">快速体验用户、容器化部署诉求人员</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">一键启动、环境隔离、无需手动配置依赖</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">想最快跑通流程，或需要多环境一致性</td>
  </tr>
</table>

### 软件包安装

#### 检查安装环境<a id = "huanjingzhunbei" ></a>
确定 CANN、Python 和 torch_npu 软件版本并安装，软件包安装和源码编译安装均需要先完成这一步。
-   CANN 版本选择：可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成 CANN 的安装与配置。建议下载安装 8.5.0 版本。

-   Python 版本选择：py3.9 不支持 aarch64。
    <table style="table-layout: fixed; width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
    <thead>
    <tr>
    <th style="width: 50%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Triton-Ascend 版本</strong>
    </th>
    <th style="width: 50%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Python支持版本</strong>
    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.1</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">py3.9-py3.13</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">py3.9-py3.11</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0rc4</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">py3.9-py3.11</td>
    </tr>
    </tbody>
    </table>

-   torch_npu 版本选择：当前配套的 torch_npu 版本为 2.7.1.post4。

#### whl包安装
1.  检查 Python 版本

    ```
    python3 --version
    ```

2.  安装 whl 包
    -    Triton-Ascend 3.2.0 及以下 Triton-Ascend 和 Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。
    -    Triton-Ascend 3.2.1 及以上，Triton-Ascend 通过将 Triton 声明为安装依赖来缓解安装覆盖问题。 安装 Triton-Ascend 时会先安装社区 Triton，再由 Triton-Ascend 覆盖同名目录，从而避免后续安装其他依赖 Triton 的软件包时再次安装 Triton 而覆盖 Triton-Ascend。
    -    X86 与 Arm 使用不同版本的社区 Triton 安装包的原因是社区从 3.5 版本开始才提供 Arm 版本安装包：X86 依赖 Triton == 3.2.0，Arm 依赖 Triton == 3.5.0。


    ```
    # 以安装 triton-ascend 3.2.1 为例
    pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
    ```


### 源码编译安装
如果需要对 **Triton-Ascend** 进行开发或自定义修改，可以采用源代码编译安装的方法。在安装环境和依赖准备好后，推荐使用[<u>在线安装</u>](#kuaisuanzhuang)的方式完成基于源码安装；若有特殊需求，如目标机器无法联网等原因，可以进行[<u>离线安装</u>](#shoudonganzhuang)。


#### 检查安装环境
确定 CANN、Python 和 torch_npu 软件版本并安装，软件包安装和源码编译安装均需要先完成这一步。具体参考软件包安装的[<u>环境准备事项</u>](#huanjingzhunbei)


**系统推荐**
**表3** PyTorch 兼容性推荐版本表
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Pytorch版本</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">推荐的GCC版本</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">推荐的GLIBC版本</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.7.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">11.2.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.8.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.9.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
    <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
</table>

#### 安装依赖
1.  安装系统库依赖
    安装 zlib1g-dev / lld / clang，可选择安装 ccache 包用于加速构建。
    -   推荐版本 clang >= 15
    -   推荐版本 lld >= 15
    ```
    sudo apt update
    sudo apt install zlib1g-dev clang-15 lld-15
    sudo apt install ccache # optional
    ```
2.  安装 Python 依赖
    ```
    pip install ninja cmake wheel pybind11 # build-time dependencies
    ```


#### 在线安装<a id = "kuaisuanzhuang" ></a>
```
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# 可选，若本地有编译好的LLVM，可以直接指定本地LLVM，不会触发下载LLVM预编译包。若无，忽略这条，直接执行下面的运行安装命令即可。
export LLVM_SYSPATH=/path/to/LLVM

# 执行安装命令
pip install -e .
```

#### 离线安装 - 基于LLVM构建<a id = "shoudonganzhuang" ></a>
Triton 使用 LLVM 22 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。LLVM 的构建支持两种构建方式，以下**两种方式二选一即可**，无需重复执行。

##### 代码准备
```
# git checkout 检出指定版本的LLVM
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout fad3272286528b8a491085183434c5ad4b59ab92
wget https://raw.githubusercontent.com/triton-lang/triton-ascend/6765b03c81c4e9ecb277e4ef1dde61dea0d044f0/third_party/ascend/llvm_patch/fad3272.patch
git apply fad3272.patch
```

##### 方式一：clang构建安装LLVM
-   步骤1：使用 clang 安装 LLVM，环境上请安装 clang、lld，并指定版本（推荐版本 clang>=15，lld>=15）， 如未安装，请按下面指令安装 clang、lld、ccache
    ```
    apt-get install -y clang-15 lld-15 ccache
    ```
-   步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径
    ```
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    ```
-   步骤3：执行以下命令进行构建和安装 LLVM
    ```
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
-   步骤4：需要拷贝 FILECHECK 到目标安装路径
    ```
    cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
    ```


##### 方式二：构建 Triton-Ascend
-   步骤1：克隆 Triton-Ascend
    ```
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    ```
-   步骤2：编译安装 Triton-Ascend
    ```
    # 确认已设置 [基于LLVM构建] 章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
    # 确认已安装clang>=15，lld>=15，ccache

    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```


### 镜像包安装
通过 Dockerfile 安装 Docker 环境镜像。使用 quay.io/ascend/cann 预构建镜像作为基础镜像，跳过 CANN 安装步骤，显著加快构建速度。

#### 检查镜像版本

**表4** CANN 版本与镜像标签对照表
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">CANN版本</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">芯片类型</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Python版本</th>
    <th style="width: 40%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">镜像标签</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-910b-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-a3-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2-910b-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2-a3-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-beta.2-a3-ubuntu22.04-py3.11</td>
  </tr>
</table>


#### 镜像安装
1.  构建镜像

    ```
    # 这里以 8.5.0-a3-ubuntu22.04-py3.10 为例
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    docker build \
    --build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
    -t triton-ascend-image:latest -f ./docker/Dockerfile .
    ```

2.  启动容器
    ```
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


## 安装结果验证
安装运行时依赖
```
# 拉取triton-ascend源码仓及用例（可选，非源码编译安装运行示例时需拉源码仓）
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend && pip install -r requirements.txt
```

运行实例：<a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>
```
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 运行tutorials示例：
python3 ./third_party/ascend/tutorials/01-vector-add.py
```
观察到类似的输出即说明环境配置正确
```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

# 附录：FAQ

## 安装 torch_npu 时出现报错“ERROR: No matching distribution found for torch==2.7.1+cpu”

### 解决措施
可以尝试手动安装 torch 后再安装 torch_npu
```
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## 编译安装 Triton-Ascend 时，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”

### 解决措施
一般是链接器无法找到 stdc++fs 库引起的报错。该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中以下相关代码片段的注释打开。
文件路径：triton-ascend/CMakeLists.txt
```
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```
