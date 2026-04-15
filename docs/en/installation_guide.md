# Installation Guide

## Preparing the Environment

### Python Version Requirements

Triton-Ascend requires Python 3.9 to 3.11.

### Installing Ascend CANN

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
| 3.2.0rc4          | CANN 8.3.RC2         | 2025-11-20        |
|                   | CANN 8.3.RC1         | 2025-10-30        |

- Community edition

| Triton-Ascend Version| CANN Community Version| CANN Release Date|
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026-01-16        |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025-11-20        |
|                   | CANN 8.5.0.alpha001  | 2025-11-12        |
|                   | CANN 8.3.RC1         | 2025-10-30        |


### Installing torch_npu

The current torch_npu version is 2.7.1.

```bash
pip install torch_npu==2.7.1
```

Note: If `ERROR: No matching distribution found for torch==2.7.1+cpu` is displayed, you can manually install Torch and then install torch_npu.
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Installing Triton-Ascend Using Pip

### Latest Stable Version
You can install the latest stable version of Triton-Ascend using pip.

```shell
pip install triton-ascend
```

- Note: Community Triton and Triton-Ascend cannot coexist. When you install other software that depends on Triton, community Triton will be automatically installed, which will overwrite the installed Triton-Ascend directory.
In this case, you need to uninstall community Triton first and then install Triton-Ascend.
```shell
pip uninstall triton
pip install triton-ascend
```

### Nightly Build Version
We provide daily updated nightly packages. You can run the following command to install them:

```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```
You can also find all nightly build packages in [History](https://test.pypi.org/project/triton-ascend/#history).

Note: If you encounter SSL-related errors when running the `pip install` command, add the `--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org` option to solve them.

## Installing Triton-Ascend Using the Source Code

If you need to develop or customize Triton-Ascend, you should install it by compiling the source code. This method allows you to adjust the source code based on project requirements and compile and install a customized Triton-Ascend version.

### System Requirements

- GCC >= 9.4.0
- GLIBC >= 2.27

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

### Building with LLVM

Triton uses LLVM 20 to generate code for GPUs and CPUs. Similarly, the BiSheng Compiler of Ascend depends on LLVM to generate NPU code. Therefore, you need to compile the LLVM source code. Pay attention to the specific LLVM version of dependencies. LLVM build supports two methods. **You only need to follow either method**.

#### Code preparation: Run the `git checkout` command to check out the specified LLVM version.

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

#### Method 1: Installing LLVM Using Clang

- Step 1: You are advised to use Clang to install LLVM. Install Clang and LLD in the environment and specify their versions (Clang >= 15 and LLD >= 15 are recommended).
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

#### Method 2: Installing LLVM Using GCC

- Step 1: Clang is recommended. However, when only GCC is available, pay attention to [Note 1](#note1) and [Note 2](#note2). Set the environment variable *LLVM_INSTALL_PREFIX* to your target installation path.

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- Step 2: Run the following commands to build and install LLVM:

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

<a id="note1"></a>Note 1: If `ld.lld: error: undefined symbol` is displayed during compilation, you can add `-DLLVM_ENABLE_LLD=ON` during Step 2.

<a id="note2"></a>Note 2: If ccache has been installed and is running properly in the environment, you can set `-DLLVM_CCACHE_BUILD=ON` to accelerate the build process. Otherwise, do not enable it.

#### Cloning Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend/python
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

- Note 3: GCC 9.4.0 or later is recommended. If the GCC version is earlier than 9.4.0, "ld.lld: error: unable to find library -lstdc++fs" may be reported, indicating that the linker cannot find the stdc++fs library.
This library supports the file system features of versions earlier than GCC 9. In this case, you need to manually uncomment the related code snippet in the CMake file.

- triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  After uncommenting the code snippet, rebuild the project to solve the problem.

2. Run the Triton example.

   Install the runtime dependencies. Refer to the following command:
   ```bash
   cd triton-ascend && pip install -r requirements_dev.txt
   ```
   Run the [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py) instance.
   ```bash
   # Set the CANN environment variables (for example, as the root user and with the default installation path /usr/local/Ascend).
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # Run the tutorials example.
   python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
   ```
    If an output similar to the following is displayed, the environment is correctly configured:
    ```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```
