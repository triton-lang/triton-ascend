# Installation Guide

**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors. It is mainly used to provide efficient kernel auto-tuning, operator compilation, and deployment capabilities, and supports the Ascend Atlas A2/A3/950 series products. While remaining compatible with core Triton syntax, it is deeply optimized for Ascend NPU features, including automatic parsing of kernel parameters, optimized memory access logic, and improved secure deployment mechanisms.

## Environment Preparation

**Hardware Requirements**

- Ascend products: Atlas A2/A3/950 series are supported.

- NPU configuration: at least 32 GB of memory per card is recommended.

- Operating system: A Linux system is required. For details, refer to the <a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">Compatibility Query Assistant</a>. All operations in the rest of this document are demonstrated in an **Ubuntu** environment.

**Software Dependencies**

Determine the CANN, Python, and TorchNPU software versions and install them. For the driver and firmware installation, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) on the official Ascend community website.

- CANN version: 9.1.0
- Python version: python3.11
- TorchNPU version: 2.7.1.post8

Note: For more compatibility relationships, refer to the [Release Notes](./release_note.md#version-compatibility-matrix).

## Quick Installation

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

<a id="install-from-source"></a>

## Source Installation

### Install Dependencies

```bash
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

### Compile Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
git checkout main
pip install -e .
```

### Custom LLVM Build (Optional)

If you need to customize the LLVM build process, follow the steps below to compile Triton-Ascend.

1. **Prepare the source code**: Check out the LLVM source code of the specified version with `git checkout` and apply the patch.

    ```bash
    git clone --no-checkout https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout f6ded0be897e2878612dd903f7e8bb85448269e5
    wget https://raw.githubusercontent.com/triton-lang/triton-ascend/refs/heads/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch
    git apply llvm_patch_f6ded0b.patch
    ```

2. **Build LLVM**: The path `{PATH_TO}` is the path where you checked out the LLVM source code in step 1.

    ```bash
    # /path/to/llvm-install is the user's planned LLVM installation path; adjust as needed
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

3. **Compile Triton-Ascend**

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

## Development Images

### Check Image Versions

**Table 2** Mapping of CANN versions to image tags.
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">CANN Version</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Chip Type</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Python Version</th>
    <th style="width: 40%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Image Tag</th>
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

### Using the Image

```bash
# Using 9.0.0-a3-ubuntu22.04-py3.11 as an example
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

# Enter the container; install Triton-Ascend via either Quick Installation or Source Installation method
docker exec -u root -it triton-ascend_container /bin/bash
```

## Running Examples

**Run the vector addition example in tutorials to verify the result**

Vector addition example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

```bash
# Set CANN environment variables (using root user default install path `/usr/local/Ascend` as example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Clone the triton-ascend repository and examples (skip if installed from source)
git clone https://github.com/triton-lang/triton-ascend.git
# Run tutorials example
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

If you see similar output, the environment is configured correctly:

```text
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## Installation FAQ

**Question 1: An error "ERROR: No matching distribution found for torch==2.7.1+cpu" is reported when installing TorchNPU**

**Solution**

You can try installing torch manually first and then installing TorchNPU:

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Question 2: When compiling and installing Triton-Ascend, if GCC < 9.4.0, the error "ld.lld: error: unable to find library -lstdc++fs" may be reported**

**Solution**

This error is usually caused by the linker being unable to find the stdc++fs library. This library is used to support the file system features of versions before GCC 9. In this case, you need to manually uncomment the following code snippet in the CMake file.
File path: triton-ascend/CMakeLists.txt

```bash
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```

**Question 3: An error ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package is reported when running operators**

**Root Cause Analysis**

The triton-ascend directory is overwritten by triton, which damages the functionality of triton-ascend.

**Solution**

Uninstall the corrupted triton-ascend and reinstall it. Taking version 3.2.1 as an example, you can run the following commands to fix the issue:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

**Question 4: Why does Triton-Ascend 3.2.1 add a dependency on triton?**

Answer: Triton-Ascend is a secondary development based on Triton and shares the same name as the Triton installation directory. If users install Triton or third-party packages that depend on Triton after installing Triton-Ascend, the Triton directory will be overwritten, which damages the functionality of Triton-Ascend.
Therefore, by adding a dependency on Triton, the following reminder will be shown when Triton is overwritten:

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```

If you encounter this issue and want to restore the functionality of Triton-Ascend, you can do the following:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

```

**Question 5: Why are the Triton versions depended on by Triton-Ascend 3.2.1 inconsistent?**

Answer: X86 and Arm use different versions of community Triton installation packages because the community has provided X86 installation packages since Triton 3.2, while Arm installation packages have been provided only since Triton 3.5.

**Question 6: How to confirm the chip type**

You can use the npu-smi command to view the NPU model on the system. For example, in the output of the npu-smi info command, "910B4" corresponds to chip type A2 (Ascend 910b series):

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
