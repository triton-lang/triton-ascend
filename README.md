<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
  </picture>
</p>

<h3 align="center">
Triton-Ascend
</h3>

<p align="center">
<a href="README.md"><b>English</b></a> | <a href="README_zh.md"><b>中文</b></a>
</p>

<p align="center">
| <a href="https://triton-ascend.readthedocs.io/en/latest/"><b>Official Documentation</b></a> | <a href="https://www.hiascend.com/developer/operator?tag=triton"><b>Operator Development User Journey</b></a> | <a href="https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR"><b>Community Meeting</b></a> | <a href="https://www.hiascend.com/"><b>About Ascend</b></a> |
</p>

---

## 🔥 Latest News

- [2026.04.30] Triton-Ascend 3.2.1 official version released
- [2026.01.20] Triton-Ascend 3.2.0 official version released

<div style="margin-left:1em">
<details>
<summary>More Latest News</summary>

- [2025.11.14] Triton-Ascend 3.2.0rc4 pre-release version released:<br>- [Extended the tt.fp_to_fp interface, adding FP8 type conversion support](https://gitcode.com/Ascend/triton-ascend/pull/891) <br>- [Added the scatter_ub_to_out interface, supporting efficient data scatter operations from UB to GM](https://gitcode.com/Ascend/triton-ascend/pull/864)
- [2025.09.30] Improved Scan/Sort Triton Python APIs, supported non-contiguous memory access, and completed adaptation of key Triton operators in the vLLM and sglang open-source repositories
- [2025.09.19] Supported Triton-Ascend [nightly package](https://test.pypi.org/project/triton-ascend/#history) extraction
- [2025.08.15] Improved Atomic Triton Python API support, completed adaptation of key Triton operators in the Flaggems open-source repository, and provided reference implementations for high-performance simple operators such as Matmul
- [2025.06.30] Supported 85% of Triton Python APIs, supported contiguous memory access, covering basic usage scenario requirements
- [2025.05.20] Triton-Ascend open-sourced, Gitcode repository alive!

</details>
</div>

---

## 📌 Introduction

Triton-Ascend is a Triton compilation framework built for the Ascend platform, designed to enable Triton code to run efficiently on Ascend hardware.
Triton-Ascend adapts and specifically optimizes the Ascend NPU through the Triton compilation stack, significantly improving functional and performance generalization capabilities.
The Triton-Ascend framework bridges Triton and Ascend hardware, lowers the development barrier, and meanwhile completes the Ascend software stack capabilities and enriches the operator and application ecosystem.

## 📖 Quick Installation

### Environment Preparation

#### Hardware Requirements

Supported operating systems: linux(aarch64/x86_64)

Supported Ascend products: Atlas A2/A3 series

Minimum hardware configuration: Single-card 32GB memory (recommended)

#### Software Dependencies

Determine and install the Python, CANN, and torch_npu software versions. Both package installation and source code compilation installation require this step to be completed first.

- Python version selection: py3.9-py3.11 are all supported.

- CANN version selection: You can visit the Ascend community official website and follow the <a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">community software installation guide</a> provided there to complete the installation and configuration of CANN. It is recommended to download and install version 9.0.0.

- torch_npu version selection: The currently matched torch_npu version is 2.7.1.post4.

## Installation via pip

### Latest Stable Version

You can install the latest stable version of Triton-Ascend via pip.

```shell
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

- Note:
For triton-ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first before installing Triton-Ascend.<br>
For triton-ascend 3.2.1 and later, Triton-Ascend mitigates the installation overwriting issue by declaring Triton as an installation dependency.<br>
When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overwrites the directory with the same name, thereby avoiding re-installing Triton and overwriting Triton-Ascend when subsequently installing other packages that depend on Triton.<br>
The reason why x86 and arm use different versions of the community Triton installation package is that the community only provides the arm version installation package from version 3.5 onwards: x86 depends on triton==3.2.0, and arm depends on triton==3.5.0.

```shell
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### Historical Stable Version

```shell
pip install triton-ascend==3.2.0
```

## Installation from Source

### Quick Installation

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# Execute the installation command
pip install -e .
```

## Build Based on LLVM

Triton uses LLVM 22 to generate code for GPU and CPU. Similarly, Ascend's BiSheng compiler also depends on LLVM to generate NPU code, so you need to compile the LLVM source code before using it. Please pay attention to the specific LLVM version of dependencies. LLVM build supports two build methods. **Choose either of the two methods below**, no need to execute both.
<div style="margin-left:1em">
<details>
<summary>More about building based on LLVM</summary>

### Code Preparation: Check out the specified version of LLVM using `git checkout`

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout fad3272286528b8a491085183434c5ad4b59ab92
   wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
   git apply fad3272.patch
   ```

### Build and Install LLVM with clang

- Step 1: Install LLVM using clang. Please install clang and lld in the environment and specify the versions (recommended versions clang>=15, lld>=15).
  If not installed, please install clang, lld, and ccache according to the following commands:

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- Step 2: Set the environment variable LLVM_INSTALL_PREFIX to your target installation path:

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- Step 3: Execute the following commands to build and install LLVM:

  ```bash
  cd {PATH_TO}/llvm_project # The path is where the user pulled the LLVM code, adjust according to actual situation
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

- Step 4: Need to copy FILECHECK to the target installation path:

   ```bash
   cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   ```

### Clone Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
```

### Build Triton-Ascend

- Step 1: Please confirm that the target installation path of LLVM ${LLVM_INSTALL_PREFIX} has been set in the [Build Based on LLVM] section
- Step 2: Please confirm that clang>=15, lld>=15, and ccache have been installed

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

Note 1: For the recommended GCC version, see the "System Recommendations" section above. If GCC < 9.4.0, the error "ld.lld: error: unable to find library -lstdc++fs" may be reported, indicating that the linker cannot find the stdc++fs library.
This library is used to support file system features for versions earlier than GCC 9. In this case, you need to manually uncomment the related code snippet in the CMake file:

triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  After uncommenting, rebuild the project to resolve this issue.
<a id="docker-build"></a>

## Installation via Dockerfile

- We provide a Dockerfile to help you install the Docker environment image. The build process uses the `quay.io/ascend/cann` pre-built image as the base image, skipping the CANN installation step and significantly speeding up the build.

- You need to specify the `CANN_BASE_IMAGE` parameter via `--build-arg` to select the CANN base image suitable for your machine. Available CANN base image tags can be found at [quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags).

- You can check the NPU model on your system using the npu-smi command.

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

- To start a container based on this image, you can refer to the following command:

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

# Enter the container
docker exec -u root -it triton-ascend_container /bin/bash
```

</details>
</div>

## 🏘️ Community Event Information

1. [Meeting Calendar](https://meeting.osinfra.cn/ascend)
2. [Meeting Minutes Board](https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR)

## 🤝 Community and Contribution

Welcome to participate in the development and code contribution of Triton-Ascend. For details, please refer to the [Contribution Guide](./CONTRIBUTING.md).

- Please use [Issue](https://github.com/triton-lang/triton-ascend/issues) to inform us of any bugs you encounter.
