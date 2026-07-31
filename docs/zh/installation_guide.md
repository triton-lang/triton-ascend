# 安装指南

**Triton-Ascend**是适配华为Ascend处理器的Triton优化版本，主要用于提供高效的核函数自动调优、算子编译及部署能力，支持Ascend Atlas A2/A3/950系列产品，兼容Triton核心语法的同时，针对昇腾NPU特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

## 环境准备

**硬件要求**

- Ascend产品：支持Atlas A2/A3/950系列。

- NPU配置：建议至少单卡32GB内存。

- 操作系统：需Linux系统，具体请参考<a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">兼容性查询助手</a>。本文接下来所有操作均以**Ubuntu**环境演示。

**软件依赖**

确定CANN、Python和TorchNPU软件版本并安装。其中，可以参考昇腾社区官网《[CANN快速安装](https://www.hiascend.com/cann/download)》
完成驱动与固件安装。

- CANN版本：9.1.0
- Python版本：python3.11
- TorchNPU版本：2.7.1.post8

注：更多配套关系请参考[版本说明表](./release_note.md#版本兼容性矩阵)。

## 快速安装

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 源码安装

### 安装依赖

```bash
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

### 编译Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
git checkout main
pip install -e .
```

### 自定义LLVM构建（可选）

如果需要自定义构建LLVM过程的，可以执行下面的步骤去编译Triton-Ascend。

1. **代码准备**：通过`git checkout`检出指定版本的LLVM源码并应用补丁。

    ```bash
    git clone --no-checkout https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout f6ded0be897e2878612dd903f7e8bb85448269e5
    wget https://raw.githubusercontent.com/triton-lang/triton-ascend/refs/heads/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch
    git apply llvm_patch_f6ded0b.patch
    ```

2. **构建LLVM**：路径`{PATH_TO}`为用户第一步检出LLVM源码的路径。

    ```bash
    # /path/to/llvm-install 路径为用户规划的llvm安装路径,需根据实际调整
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    cd {PATH_TO}/llvm-project
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

3. **编译Triton-Ascend**

    ```bash
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```

## 开发镜像

### 检查镜像版本

**表2** CANN版本与镜像标签对照表。
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
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-950-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-910b-ubuntu22.04-py3.12</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-a3-ubuntu22.04-py3.12</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-950-ubuntu22.04-py3.12</td>
  </tr>
</table>

### 镜像使用

```bash
# 这里以 9.0.0-a3-ubuntu22.04-py3.11 为例
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container \
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
quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11 \
/bin/bash

# 进入容器，可在前面的快速安装和源码安装中任选一种方式安装Triton-Ascend
docker exec -u root -it triton-ascend_container /bin/bash
```

## 运行样例

**运行tutorials中向量加法实例验证结果**

向量加法实例：<a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取triton-ascend源码仓及用例（使用源码安装Triton-Ascend的无需重复拉取）
git clone https://github.com/triton-lang/triton-ascend.git
# 运行tutorials实例
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确：

```text
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## 安装常见问题

**问题一：安装TorchNPU时出现报错“ERROR: No matching distribution found for torch==2.7.1+cpu”**

**解决措施**

可以尝试手动安装torch后再安装TorchNPU：

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

**问题二：编译安装Triton-Ascend时，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”**

**解决措施**

一般是链接器无法找到stdc++fs库引起的报错。该库用于支持GCC 9之前版本的文件系统特性。此时需要手动把CMake文件中以下相关代码片段的注释打开。
文件路径：triton-ascend/CMakeLists.txt

```bash
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```

**问题三：执行算子时报错 ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package**

**根因分析**

 triton-ascend目录被triton覆盖,导致triton-ascend功能受损。

**解决措施**

 卸载已损坏的triton-ascend,重新安装即可。以3.2.1 版本为例，可执行如下命令修复：

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

**问题四：Triton-Ascend 3.2.1版本为何新增依赖triton？**

答复：Triton-Ascend是基于Triton进行的二次开发，与Triton安装目录同名。若用户安装Triton-Ascend之后，在此安装Triton或依赖Triton的三方件，会覆盖Triton目录，导致Triton-Ascend功能受损。
因此通过增加Triton依赖，当Triton被覆盖安装时会有如下提醒。

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```

若用户遇到且想恢复Triton-Ascend功能，可做如下操作：

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

```

**问题五：Triton-Ascend 3.2.1版本依赖的Triton版本为何不一致？**

答复：X86与Arm使用不同版本的社区Triton安装包，是因为社区从Triton 3.2版本开始提供X86安装包，而Arm安装包是从Triton 3.5版本开始提供的。

**问题六：如何确认芯片类型**

您可以使用npu-smi命令查看系统上的NPU型号。例如，在npu-smi info命令的输出中，"910B4" 对应芯片类型A2（昇腾910b系列）：

```Text
root@localhost:/# npu-smi  info
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B4               | OK            | 82.6                 32                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                2871 / 32768            |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0                                                                              |
```
