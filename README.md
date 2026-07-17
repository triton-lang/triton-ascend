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
| <a href="https://triton-ascend.readthedocs.io/zh-cn/latest/"><b>Official Documentation</b></a> | <a href="https://www.hiascend.com/developer/operator?tag=triton"><b>Operator Development User Journey</b></a> | <a href="https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR"><b>Community Meetings</b></a> | <a href="https://www.hiascend.com/"><b>About Ascend</b></a> |
</p>

---

## 🔥 Latest News

- [2026.04.30] Triton-Ascend 3.2.1 official release is now available
- [2026.01.20] Triton-Ascend 3.2.0 official release is now available

<div style="margin-left:1em">
<details>
<summary>More latest news</summary>

- [2025.11.14] Triton-Ascend 3.2.0rc4 pre-release is now available:<br>- [Extended the tt.fp_to_fp interface to add FP8 type conversion support](https://gitcode.com/Ascend/triton-ascend/pull/891) <br>- [Added the scatter_ub_to_out interface to support efficient data scatter operations from UB to GM](https://gitcode.com/Ascend/triton-ascend/pull/864)
- [2025.09.30] Improved Scan/Sort Triton Python APIs, supporting non-contiguous memory access, and completed adaptation of key Triton operators in vLLM and sglang open-source repositories
- [2025.09.19] Supported Triton-Ascend [nightly package](https://test.pypi.org/project/triton-ascend/#history) extraction
- [2025.08.15] Improved Atomic-class Triton Python API support, completed adaptation of key Triton operators in the Flaggems open-source repository, and provided reference examples for high-performance implementations of simple operators such as Matmul
- [2025.06.30] Supported 85% of Triton Python APIs, supporting contiguous memory access, covering basic usage scenarios
- [2025.05.20] Triton-Ascend is open-sourced, Gitcode repository is alive!

</details>
</div>

---

## 📖 Quick Installation

### Environment Preparation

#### Hardware Requirements

Supported operating systems: linux (aarch64/x86_64)

Supported Ascend products: Atlas A2/A3/950 series

Minimum hardware configuration: single card with 32GB memory (recommended)

#### Software Dependencies

Determine and install the Python, CANN, and TorchNPU software versions. This step must be completed before both package installation and source code compilation installation.

- Python version selection: py3.9-py3.11 are all supported.

- CANN version selection: You can visit the Ascend community website and follow the <a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">community software installation guide</a> to complete the CANN installation and configuration. It is recommended to download and install version 9.0.0.

- TorchNPU version selection: The currently bundled TorchNPU version is 2.7.1.post4.

### Quick Installation

```bash
# Taking the installation of triton-ascend 3.2.1 as an example
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

### Source Installation

<div style="margin-left:1em">
<details>
<summary>More source installation</summary>

#### Install Dependencies

```bash
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

#### Build Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
git checkout main
pip install -e .
```

#### Custom LLVM Build (Optional)

```bash
# If you need to customize the LLVM build process, you can execute this step first before compiling Triton-Ascend
# Check out the specified version of LLVM source code and apply patches
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout fad3272286528b8a491085183434c5ad4b59ab92
wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
git apply fad3272.patch

export LLVM_INSTALL_PREFIX=/path/to/llvm-install

# Build a custom LLVM version
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

# Compile Triton-Ascend
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

### Docker Image Usage

<div style="margin-left:1em">
<details>
<summary>More Docker image usage</summary>

- We provide a Dockerfile to help you install the Docker environment image. The build process uses the `quay.io/ascend/cann` pre-built image as the base image, skipping the CANN installation step and significantly speeding up the build.

- You need to specify the `CANN_BASE_IMAGE` parameter via `--build-arg` to select the appropriate CANN base image for your machine. Available CANN base image tags can be found at [quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags).

- You can check the NPU model on your system using the npu-smi command.

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

- To start a container from this image, you can refer to the following command:

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

## ✏️ Documentation Entry

- [Quick Start](./docs/zh/quick_start.md)

- [Complete Online Documentation (Recommended)](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)

- [Installation Guide](./docs/zh/installation_guide.md)

- [Architecture Design and Core Features](./docs/zh/architecture_design_and_core_features.md)

- [Operator Development Guide](./docs/zh/programming_guide/index.md)

- [Operator Migration Guide](./docs/zh/migration_guide/migrate_from_gpu.md)

- [Operator Debugging Guide](./docs/zh/debug_guide/debugging.md#)

- [Performance Tuning Guide](./docs/zh/debug_guide/profiling.md#)

- [Environment Variables Reference](./docs/zh/environment_variable_and_compiler_options_reference.md)

- [FAQ](./docs/zh/FAQ.md)

## 🏘️ Community Activities

1. [Meeting Calendar](https://meeting.osinfra.cn/ascend)
2. [Meeting Minutes Board](https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR)

## 🤝 Community and Contribution

- Welcome to participate in Triton-Ascend development and code contribution. For details, please refer to the [Contribution Guide](./docs/zh/community/CONTRIBUTING_zh.md)

- Please report any bugs you encounter via [Issue](https://github.com/triton-lang/triton-ascend/issues).
