# Quick Installation

This article mainly introduces how to quickly complete the installation of **Triton-Ascend** basic supporting components in an Ubuntu environment. For detailed operation steps, please refer to [<u>Installation Guide</u>](#installation-guide).

## Quick Setup Based on Docker Image

Directly use the out-of-the-box images released by Triton-Ascend to quickly build a development environment.

### Confirm Image

**Table 1** Partial mapping table of Ascend chips, corresponding products, and image tags. For more images, please refer to the [OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md) document.
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Chip Model</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Corresponding Product</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Image Tag</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend 910b</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A2, Atlas 900 A2 PoD</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend 950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 950PR Series</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-950-ubuntu22.04-py3.11</td>
  </tr>
</table>

### Implementation

1. Create container

    ```bash
    # Assume your NPU device model is A3, the device is installed at /dev/davinci1, and your NPU driver is installed at /usr/local/Ascend:
    # Take image_tag: 3.2.1-a3-ubuntu22.04-py3.11 as an example:
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

2. Enter container

    ```bash
    docker exec -it triton-ascend_container bash
    ```

3. Pull code

    ```bash
    # Pull triton-ascend source code repository and examples
    git clone https://github.com/triton-lang/triton-ascend.git
    cd triton-ascend
    ```

4. Run example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

    ```bash
    # Run tutorials example:
    python3 ./third_party/ascend/tutorials/01-vector-add.py
    ```

    Observing similar output indicates that the environment has been set up successfully:

    ```text
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```

# Installation Guide <a id="installation-guide"></a>

## Overview

Triton-Ascend is an optimized version of Triton adapted for Huawei Ascend chips. It is mainly used to provide efficient kernel auto-tuning, operator compilation, and deployment capabilities. It supports Ascend Atlas A2/A3 and other series products. While being compatible with Triton's core syntax, it has been deeply optimized for Ascend NPU characteristics, including automatic parsing of kernel parameters, optimization of memory access logic, and improvement of secure deployment mechanisms.

This article mainly introduces three installation methods for Triton-Ascend: package installation; image installation; source code compilation installation.

## Hardware and Operating System

- Ascend products: Support Atlas A2/A3/A5 series.
- NPU configuration: At least single-card 32GB memory is recommended.
- Operating system: Linux system is required. For specific selection, please refer to <a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">Compatibility Query Assistant</a>. All subsequent operations in this article are demonstrated in the Ubuntu environment.

## Installation Method Selection

Quick decision: Most users can directly choose package-based installation; choose image installation for containerized deployment; choose source code compilation installation for secondary development and code modification.
**Table 2** Comparison table of differences among installation methods
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 24.41%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Installation Method</th>
    <th style="width: 19.15%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Target Users</th>
    <th style="width: 26.21%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Core Advantages</th>
    <th style="width: 30.23%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Selection Reason</th>
  </tr>
    <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Package Installation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Production environment users, O&M personnel</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Simple installation, automatic dependency management, convenient upgrade and uninstall</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Pursuing stability, rapid deployment, no environment hassle</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Source Code Compilation Installation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Developers, users needing custom features</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">High customizability, support for latest features</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Need to modify source code, adapt to special hardware or features</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Image Installation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Quick experience users, containerized deployment personnel</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">One-click startup, environment isolation, no manual dependency configuration</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Want to run through the process fastest, or need multi-environment consistency</td>
  </tr>
</table>

### Package Installation

#### Related Product Versions

<table style="table-layout: fixed; width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
    <thead>
    <tr>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Triton-Ascend Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Supported Python Versions</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>CANN Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Torch-NPU Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Remarks</strong>
    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.1</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x<br>Python3.12.x<br>Python3.13.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">9.0.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.7.1.post4<br>2.8.0.post4<br>2.9.0.post2<br>2.10.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x does not support aarch64</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">8.5.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.6.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">NA</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0rc4</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">8.5.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.6.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">NA</td>
    </tr>
    </tbody>
    </table>

#### Check Installation Environment<a id="env-prepare"></a>

Determine and install the CANN, Python, and Torch-NPU software versions. Both package installation and source code compilation installation require this step to be completed first.
- Recommended CANN version: 9.0.0
- Recommended Python version: python3.11
- Recommended PyTorch version: 2.7.1
- Recommended Torch-NPU version: 2.7.1.post4

#### whl Package Installation

1. Check Python version

    ```bash
    python3 --version
    ```

    If the command output is as follows, it indicates that the Python version is 3.11.15:

    ```text
    root@test:/# python3 --version
    Python 3.11.15
    ```

2. Install whl package
    - For Triton-Ascend 3.2.0 and earlier versions, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first before installing Triton-Ascend.
    - For Triton-Ascend 3.2.1 and later versions, Triton-Ascend mitigates the installation overwriting issue by declaring Triton as an installation dependency. For details, see [FAQ](#appendix-faq)

    ```bash
    # Take installing triton-ascend 3.2.1 as an example
    pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
    ```

### Source Code Compilation Installation

If you need to develop or customize **Triton-Ascend**, you can use the source code compilation installation method. After the installation environment and dependencies are ready, it is recommended to use the [<u>Online Installation</u>](#quick-install) method to complete the source-based installation; if you have special requirements, such as the target machine cannot connect to the network, you can perform [<u>Offline Installation</u>](#manual-install).

#### Check Installation Environment

Determine and install the CANN, Python, and torch_npu software versions. Both package installation and source code compilation installation require this step to be completed first. For details, please refer to the [<u>Check Installation Environment</u>](#env-prepare) in the package installation section.

**System Recommendations**
**Table 3** PyTorch compatibility recommended version table
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">PyTorch Version</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Recommended GCC Version</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Recommended GLIBC Version</th>
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

#### Install Dependencies

1. Install system library dependencies
    Install zlib1g-dev / lld / clang. You can optionally install the ccache package to accelerate the build.
    - Recommended version clang >= 15
    - Recommended version lld >= 15

    ```bash
    apt update
    apt install zlib1g-dev clang-15 lld-15
    apt install ccache # optional
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
    ```

2. Install Python dependencies

    ```bash
    pip install ninja cmake wheel pybind11 # build-time dependencies
    ```

#### Online Installation<a id="quick-install"></a>

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# Optional: If you have a pre-compiled LLVM locally, you can specify the local LLVM directly, which will not trigger downloading the LLVM pre-compiled package. If not, ignore this line and directly execute the installation command below.
export LLVM_SYSPATH=/path/to/LLVM

# Execute the installation command
pip install -e .
```

#### Offline Installation - Build Based on LLVM<a id="manual-install"></a>

Triton uses LLVM 22 to generate code for GPU and CPU. Similarly, Ascend's BiSheng compiler also depends on LLVM to generate NPU code, so you need to compile the LLVM source code before using it. Please pay attention to the specific LLVM version of dependencies.

Prerequisite: the steps in this section run from the root of a `triton-ascend` checkout, and that checkout must already exist on this machine. If the target machine is offline, clone the repository on a machine with network access and copy the whole checkout over (for example with `scp -r` or `tar`) before continuing; if it is online, clone it now and enter the repository root:

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
```

The "offline" in this section's title refers to installing without downloading any prebuilt package: every component is built from source on this machine. Obtaining the source trees themselves — the clone below and the LLVM checkout in [Code Preparation](#code-preparation) — still needs network access, so transfer them to the offline machine together with the `triton-ascend` checkout.

##### Code Preparation

Check out the specified version of LLVM source code using `git checkout` and apply the patch:

```bash
# Run from the root of your triton-ascend checkout so that cmake/llvm-hash.txt
# resolves; the file must contain exactly one valid LLVM commit hash.
TRITON_ROOT="$(pwd)"
[ -f "$TRITON_ROOT/cmake/llvm-hash.txt" ] || { echo "Run this block from the root of your triton-ascend checkout: cmake/llvm-hash.txt not found under $(pwd)."; exit 1; }
LLVM_COMMIT=$(cat "$TRITON_ROOT/cmake/llvm-hash.txt" | tr -d '\r\n') || exit 1
# The pin must be exactly one 40-hex-character commit hash; an empty or
# malformed file would otherwise fail later inside git checkout.
if [ "${#LLVM_COMMIT}" -ne 40 ] || [ -n "$(printf '%s' "$LLVM_COMMIT" | tr -d '0-9a-f')" ]; then
  echo "cmake/llvm-hash.txt is empty or not a single 40-character commit hash — please check the file." >&2
  exit 1
fi
git clone --no-checkout https://github.com/llvm/llvm-project.git || exit 1
cd llvm-project
git checkout "$LLVM_COMMIT" || exit 1
# Force the C locale so git apply behaves deterministically regardless of the
# shell session's locale settings.
export LC_ALL=C
# The filename token between llvm_patch_ and .patch may embed the target
# commit as a hex prefix. A hex-only token that is not a prefix of the
# commit pinned in llvm-hash.txt marks a patch left over from a different
# LLVM revision: reject it here with a clear message instead of letting
# git apply fail later with an obscure error. Tokens that are not hex-only
# are descriptive names and carry no commit reference. Uppercase hex digits
# are normalized to lowercase first, so a patch file named with the
# uppercase spelling of the pinned commit's prefix is still accepted.
for patch in "$TRITON_ROOT"/third_party/ascend/patch/llvm_patch_*.patch; do
  [ -f "$patch" ] || continue
  name=$(basename "$patch")
  token=${name#llvm_patch_}
  token=${token%.patch}
  token=$(printf '%s' "$token" | tr 'A-F' 'a-f')
  case "$token" in
    ""|*[!0-9a-f]*) ;;
    *)
      case "$LLVM_COMMIT" in
        "$token"*) ;;
        *)
          echo "Patch $name targets a different LLVM revision than the commit pinned in llvm-hash.txt ($LLVM_COMMIT) — update the patch and the pin together." >&2
          exit 1
          ;;
      esac
      ;;
  esac
  git apply "$patch" || exit 1
done
```

##### Build and Install LLVM

- Step 1: Set the environment variable LLVM_INSTALL_PREFIX to your target installation path

    ```bash
    # The path is the user-planned LLVM installation path, adjust according to actual situation
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    ```

- Step 2: Execute the following commands to build and install LLVM

    ```bash
    # The following steps assume the Code Preparation block above ran in this
    # same shell. Run this block standalone from the triton-ascend root.
    # Step 1's LLVM_INSTALL_PREFIX is also shell-local: in a fresh shell,
    # re-run Step 1 (or re-export it) before this block.
    [ -n "$LLVM_INSTALL_PREFIX" ] || { echo "LLVM_INSTALL_PREFIX is not set — run Step 1 to set the target installation path."; exit 1; }
    TRITON_ROOT="${TRITON_ROOT:-$(pwd)}"
    [ -d "$TRITON_ROOT/llvm-project" ] || { echo "llvm-project/ not found under $TRITON_ROOT — run the Code Preparation block first to clone it."; exit 1; }
    cd "$TRITON_ROOT/llvm-project"
    mkdir -p build
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

- Step 3: Need to copy FILECHECK to the target installation path

    ```bash
    # Run from the triton-ascend root, or keep the same shell as Step 2
    # (which exports TRITON_ROOT); running from the llvm-project/build
    # directory without TRITON_ROOT set would resolve the wrong path.
    [ -n "$LLVM_INSTALL_PREFIX" ] || { echo "LLVM_INSTALL_PREFIX is not set — run Step 1 to set the target installation path."; exit 1; }
    TRITON_ROOT="${TRITON_ROOT:-$(pwd)}"
    [ -f "$TRITON_ROOT/llvm-project/build/bin/FileCheck" ] || { echo "FileCheck not found under $TRITON_ROOT — run Steps 1-2 from the triton-ascend root first."; exit 1; }
    cp  "$TRITON_ROOT"/llvm-project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
    ```

##### Build Triton-Ascend

- Step 1: Clone Triton-Ascend

    The prerequisite at the start of this section already cloned the
    repository and entered its root, so this step needs no commands. If you
    are unsure whether you are still there, Step 2's first command returns
    to the repository root before building.

- Step 2: Compile and install Triton-Ascend

    ```bash
    # Return to the repository root: the LLVM build steps above leave the
    # working directory inside llvm-project/build, where setup.py does not exist.
    # TRITON_ROOT falls back to the current directory for a fresh shell run
    # from the repository root.
    TRITON_ROOT="${TRITON_ROOT:-$(pwd)}"
    [ -f "$TRITON_ROOT/setup.py" ] || { echo "setup.py not found under $TRITON_ROOT — run this block from the triton-ascend repository root."; exit 1; }
    [ -n "$LLVM_INSTALL_PREFIX" ] || { echo "LLVM_INSTALL_PREFIX is not set — run the [Build Based on LLVM] steps to set the target installation path first."; exit 1; }
    cd "$TRITON_ROOT"
    # Confirm that the target installation path of LLVM ${LLVM_INSTALL_PREFIX} has been set in the [Build Based on LLVM] section
    # Confirm that clang>=15, lld>=15, and ccache have been installed

    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```

### Image Installation

Install the Docker environment image through Dockerfile. Use the quay.io/ascend/cann pre-built image as the base image, skip the CANN installation step, and significantly speed up the build.

#### Check Image Version

**Table 4** CANN version and image tag mapping table
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

#### Image Installation

1. Build image

    ```bash
    # Here we take 9.0.0-a3-ubuntu22.04-py3.11 as an example
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    docker build \
    --build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11 \
    -t triton-ascend-image:latest -f ./docker/Dockerfile .
    ```

2. Start container

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

    # Enter container
    docker exec -u root -it triton-ascend_container /bin/bash
    ```

## Installation Result Verification

Install runtime dependencies:

```bash
# Pull triton-ascend source code repository and examples (optional; required to pull the source code repository when running examples without source code compilation installation)
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend && pip install -r requirements.txt
```

Run example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

```bash
# Set CANN environment variables (taking root user default installation path `/usr/local/Ascend` as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Run tutorials example:
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

Observing similar output indicates that the environment is configured correctly:

```text
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

# Appendix: FAQ<a id="appendix-faq"></a>

## When installing torch_npu, the error "ERROR: No matching distribution found for torch==2.7.1+cpu" appears

### Solution

You can try manually installing torch before installing torch_npu:

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## When compiling and installing Triton-Ascend, if GCC < 9.4.0, the error "ld.lld: error: unable to find library -lstdc++fs" may be reported

### Solution

This error is generally caused by the linker being unable to find the stdc++fs library. This library is used to support file system features for versions earlier than GCC 9. In this case, you need to manually uncomment the following related code snippet in the CMake file.
File path: triton-ascend/CMakeLists.txt

```cmake
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```

## When running an operator, the error "ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package" is reported

### Root Cause Analysis

The triton-ascend directory is overwritten by triton, causing triton-ascend functionality to be damaged.

### Solution

Uninstall the damaged triton-ascend and reinstall it. Taking version 3.2.1 as an example, you can run the following command to fix it:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

## Why does Triton-Ascend 3.2.1 add a dependency on triton?

Answer: Triton-Ascend is a secondary development based on Triton, and shares the same installation directory name with Triton. If users install Triton-Ascend and then install triton or third-party packages that depend on triton, the triton directory will be overwritten, causing Triton-Ascend functionality to be damaged.
Therefore, by adding the triton dependency, when triton is overwritten and installed, the following reminder will appear.

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```

If users encounter this and want to restore triton-ascend functionality, they can do the following:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

```

## Why are the Triton versions that Triton-Ascend 3.2.1 depends on inconsistent?

Answer: The reason why x86 and arm use different versions of the community Triton installation package is that the community only provides the arm version installation package from version 3.5 onwards: x86 depends on triton==3.2.0, and arm depends on triton==3.5.0.

## How to Confirm Chip Type

You can use the npu-smi command to check the NPU model on your system. For example, in the output of the npu-smi info command, "910B4" corresponds to chip type A2 (Ascend 910b series):

```text
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
