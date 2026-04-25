# Installation Guide

Choose the appropriate installation method as needed and proceed to the corresponding steps:

- **Install via pip**: Select this option if you intend to use the pip package of Triton-Ascend directly. Please proceed to the next step to complete the prerequisite configuration in <a href="#env-prepare">Environment Preparation</a>, then perform the pip installation.
- **Install from source**: Select this option for developers working with the Triton-Ascend source code. Please proceed to the next step to complete the prerequisite configuration in <a href="#env-prepare">Environment Preparation</a>, then choose either <a href="#auto-code-base">Quick Installation</a> or <a href="#hand-code-base">Manual Installation</a>.
- **Install via Docker**: No environment preparation required. You may directly proceed to <a href="#docker-build">Build with Docker</a>.

<a id="env-prepare"></a>

## Preparing the Environment

### Python Version Requirements

Triton-Ascend requires Python 3.9 to 3.11.

### Installing CANN

Compute Architecture for Neural Networks (CANN) is a heterogeneous compute architecture developed by Ascend for AI scenarios.
It plays a pivotal bridging role: providing upward integration with multiple AI frameworks (including MindSpore, PyTorch, and TensorFlow), while offering downward support for AI processors and programming. This establishes it as a key platform for improving the computing efficiency of Ascend AI processors.

You can visit the Ascend community website, and install and configure CANN according to the provided [software installation guide](https://www.hiascend.com/cann/download). Developers can select the CANN version, product series, CPU architecture, operating system, and installation method to find the corresponding installation commands.

During the installation, select one of the following CANN versions in *{version}*. It is advisable to download and install version 8.5.0.

- Note: If the installation path is not specified, software will be installed in the default path. The default installation paths are as follows: For the **root** user, the path is `/usr/local/Ascend`. For non-root users, the path is `${HOME}/Ascend`, where `${HOME}` indicates the current user's directory.
The preceding environment variable configurations take effect only in the current window. You can add the `source ${HOME}/Ascend/ascend-toolkit/set_env.sh` command to the environment variable configuration file (such as the .bashrc file) as required.

**CANN version:**

- Commercial edition

| Triton-Ascend Version| CANN Commercial Version| CANN Release Date|
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026-01-16        |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.3.RC1         | 2025/11/20<br>2025/10/30         |

- Community edition

| Triton-Ascend Version| CANN Community Version| CANN Release Date|
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026-01-16        |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.5.0.alpha001<br>CANN 8.3.RC1         | 2025/11/20<br>2025/11/12<br>2025/10/30         |

### Installing torch_npu

The current torch_npu version is 2.7.1.

```bash
pip install torch_npu==2.7.1
```

Note: If `ERROR: No matching distribution found for torch==2.7.1+cpu` is displayed, you can manually install Torch and then install torch_npu.

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

<a id="pip-base"></a>

## Installing Triton-Ascend Using Pip

### Latest Stable Version

You can install the latest stable version of Triton-Ascend using pip.

```shell
pip install triton-ascend
```

- Note: Community Triton and Triton-Ascend cannot coexist. When you install other software that depends on Triton, community Triton will be automatically installed, which will overwrite the installed Triton-Ascend directory.

In this case, you need to uninstall community Triton and Triton-Ascend first, and then install Triton-Ascend.

```shell
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

### Nightly Build Version

We provide daily updated nightly packages. You can run the following command to install them:

```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```

You can also find all nightly build packages in [History](https://test.pypi.org/project/triton-ascend/#history).

Note: If you encounter SSL-related errors when running the `pip install` command, add the `--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org` option to solve them.

<a id="code-base"></a>

## Installing Triton-Ascend Using the Source Code

If you need to develop or customize Triton-Ascend, you should install it by compiling from source. This method allows you to modify the source code according to your project requirements and build a customized version of Triton-Ascend.

Before building, you need to install the required build dependencies as outlined in <a href="#code-require">Dependency Installation</a>.

We recommend completing the source installation of Triton-Ascend using the <a href="#auto-code-base">Quick Installation</a> method. If you have special requirements, such as no network access on the target machine, you can use <a href="#hand-code-base">Manual Installation</a> instead.

### System Requirements

| Pytorch Version | Recommended GCC version | Recommended GLIBC version |
|-------------------|----------------------|--------------------|
| PyTorch2.6.0      | (aarch64)11.2.1<br>(x86) 9.3.1 | (aarch64)>=2.28<br>(x86)>=2.17 |
| PyTorch2.7.1      | 11.2.1               | 2.28               |
| PyTorch2.8.0      | 13.3.1               | 2.28               |
| PyTorch2.9.1      | 13.3.1               | 2.28               |
| PyTorch2.10       | 13.3.1               | 2.28               |

<a id="code-require"></a>

### Dependencies

#### Installing System Library Dependencies

Install zlib1g-dev, LLD and Clang. You can also install ccache to accelerate the build process.

- Recommended version: Clang >= 15
- Recommended version: LLD >= 15

```bash
Taking Ubuntu as an example:
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache # optional
```

Triton-Ascend depends heavily on zlib1g-dev. If you use the yum source, run the following installation command:

```bash
sudo yum install -y zlib-devel
```

#### Installing Python Dependencies

```bash
pip install ninja cmake wheel pybind11 # build-time dependencies
```

<a id="auto-code-base"></a>

### Quick Installation

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git
cd triton-ascend
git checkout main

# Optional: If a pre-compiled LLVM is available locally, you can specify the path to avoid downloading the pre-built LLVM package.
# Skip this command if no local LLVM exists and execute the installation command directly.
export LLVM_SYSPATH=/path/to/LLVM

# Run the installation command
pip install -e python
```

<a id="hand-code-base"></a>

### Manual Installation - Building with LLVM

Triton uses LLVM 22 to generate code for GPUs and CPUs. Similarly, the BiSheng Compiler of Ascend depends on LLVM to generate NPU code. Therefore, you need to compile the LLVM source code. Pay attention to the specific LLVM version of dependencies. LLVM build supports two methods. **You only need to follow either method**.

#### Code preparation: Run the `git checkout` command to check out the specified LLVM version

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout fad3272286528b8a491085183434c5ad4b59ab92
   wget https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch
   git apply fad3272.patch
   ```

#### Installing LLVM Using Clang

- Step 1: We use Clang to install LLVM. Install Clang and LLD in the environment and specify their versions (Clang >= 15 and LLD >= 15 are recommended).
  If Clang, LLD, and ccache are not installed, run the following commands to install them:

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- Step 2: Set the environment variable *LLVM_INSTALL_PREFIX* to your target installation path.

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- Step 3: Run the following commands to build and install LLVM:

  ```bash
  cd $HOME/llvm-project # Path to the LLVM code pulled by git clone
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

- Step 4: Need to cp FILECHECK to your target installation path:

   ```bash
   cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   ```

#### Cloning Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
```

#### Building Triton-Ascend

1. Install the source code.

   - Step 1: Ensure that the target installation path of LLVM (*${LLVM_INSTALL_PREFIX}*) has been set in the [Building with LLVM] section.
   - Step 2: Ensure that Clang 15 or later, LLD 15 or later, and ccache have been installed.

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

   Note 1: For the recommended GCC version, please refer to the earlier section "System Requirements". If the GCC version is earlier than 9.4.0, "ld.lld: error: unable to find library -lstdc++fs" may be reported, indicating that the linker cannot find the stdc++fs library.
   This library supports the file system features of versions earlier than GCC 9. In this case, you need to manually uncomment the related code snippet in the CMake file.

   triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

   After uncommenting the code snippet, rebuild the project to solve the problem.

<a id="docker-build"></a>

## Installation via Docker

We provide a Dockerfile to help you build a Docker environment image. The build uses pre-built CANN images from `quay.io/ascend/cann` as the base, which significantly speeds up the build process by skipping the CANN installation step.

You need to specify the `CANN_BASE_IMAGE` build arg to select the appropriate CANN base image for your machine. Available CANN base image tags can be found at [quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags).

| CANN Version | Chip Type | Python | Image Tag |
|---|---|---|---|
| 8.5.0 | 910b | 3.10 | `8.5.0-910b-ubuntu22.04-py3.10` |
| 8.5.0 | A3 | 3.10 | `8.5.0-a3-ubuntu22.04-py3.10` |
| 8.5.0 | 910b | 3.11 | `8.5.0-910b-ubuntu22.04-py3.11` |
| 8.5.0 | A3 | 3.11 | `8.5.0-a3-ubuntu22.04-py3.11` |
| 9.0.0-beta.2 | 910b | 3.10 | `9.0.0-beta.2-910b-ubuntu22.04-py3.10` |
| 9.0.0-beta.2 | A3 | 3.10 | `9.0.0-beta.2-a3-ubuntu22.04-py3.10` |
| 9.0.0-beta.2 | 910b | 3.11 | `9.0.0-beta.2-910b-ubuntu22.04-py3.11` |
| 9.0.0-beta.2 | A3 | 3.11 | `9.0.0-beta.2-a3-ubuntu22.04-py3.11` |

You can check the NPU model on your system using the `npu-smi` command.

For the machines corresponding to different chip types, refer to the table below:

| Option No. | **Chip Type** | Corresponding Server/Product Series | Typical Server Model |
|:----------:|:-------------------:|:----------------------------------:|:-----------------------------------:|
| 1 | `A3` | Atlas A3 Training Series | Atlas 900 A3 SuperPoD |
| 2 | `A2` | Atlas A2 Training Series | Atlas 800T A2 |

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

To start a container from this image, you can use the following command as a reference:

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

## Run the Triton example

   Install the runtime dependencies. Refer to the following command:

 ```bash
   # Pull the triton-ascend source code repository and examples (optional; required to pull the source code repository when running examples without source code compilation and installation).
   git clone https://gitcode.com/Ascend/triton-ascend.git
   cd triton-ascend && pip install -r requirements_dev.txt
 ```

   Run the [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py) instance.

 ```bash
   # Set the CANN environment variables (for example, as the root user and with the default installation path /usr/local/Ascend).
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # Run the tutorials example.
   python3 ./third_party/ascend/tutorials/01-vector-add.py
 ```

  If an output similar to the following is displayed, the environment is correctly configured:

```python
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
```
