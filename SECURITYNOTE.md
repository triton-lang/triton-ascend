# Triton-Ascend Security Note

## System Security Hardening

You are advised to enable the address space layout randomization (ASLR) (level 2) in the system. You can perform the following operation to enable it:

    echo 2 > /proc/sys/kernel/randomize_va_space

## Suggestions on Running Users

To ensure security and minimize permissions, you are advised not to use administrator accounts such as **root**.

## File Permission Control

1. You are advised to take security measures such as permission control on sensitive files, such as personal privacy data and business assets. For details about how to set the permissions, see the "File Permission Reference" section.

2. During the installation and use, you are advised to control the permissions. For details about how to set the permissions, see [File Permission Reference](#file-permission-reference).

### File permission reference

|   Type                            |   Maximum Permission in Linux  |
|----------------------------------- |-----------------------|
|  Home directory                        |   750 (rwxr-x---)    |
|  Program files (including scripts and libraries)    |   550 (r-xr-x---)    |
|  Program file directory                      |   550 (r-xr-x---)    |
|  Configuration files                          |   640 (rw-r-----)    |
|  Configuration file directory                      |   750 (rwxr-x---)    |
|  Log files (recorded or archived)    |   440 (r--r-----)    |
|  Log files (being recorded)               |   640 (rw-r-----)   |
|  Log file directory                      |   750 (rwxr-x---)    |
|  Debug files                        |   640 (rw-r-----)     |
|  Debug file directory                     |   750 (rwxr-x---)    |
|  Temporary file directory                      |   750 (rwxr-x---)    |
|  Maintenance and upgrade file directory                  |   770 (rwxrwx---)     |
|  Service data files                      |   640 (rw-r-----)     |
|  Service data file directory                  |   750 (rwxr-x---)    |
|  Key component, private key, certificate, and ciphertext file directory  |   700 (rwx------)      |
|  Key components, private keys, certificates, and ciphertext files      |   600 (rw-------)    |
|  APIs and script files for encryption and decryption             |   500 (r-x------)     |

## Build Security Statement

Triton-Ascend can be installed through source code compilation. During the compilation process, dependent third-party libraries are downloaded and the shell build script is executed. This results in the generation of temporary program files and compilation directories. You can control permissions on files in the source code directory as required to prevent security risks.

## Public IP Address Statement

Public IP addresses are used in the configuration files and scripts of Triton-Ascend. For details, see the "Public IP Addresses" section.

### Public IP addresses

| Type    | Open-Source Code Address                                                                                    | File Name                                     | Public IP Address/Public URL/Domain Name/Email Address                                                                | Description                         |
|----------|------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| Introduced by open source | <https://gitcode.com/Ascend/AscendNPU-IR.git> | .gitmodules | <https://gitcode.com/Ascend/AscendNPU-IR.git> | AscendNPU IR source code repository. |
| Self-developed | N/A | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/> | CANN toolkit and operator packages: Ascend-cann-{toolkit,910b-ops,950-ops,A3-ops}_{version}_linux-${ARCH}.run. |
| Introduced by open source | <https://mirrors.ustc.edu.cn> | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://mirrors.ustc.edu.cn/pypi/simple> | Default Python pip source (PIP_SOURCE). |
| Introduced by open source | <https://www.python.org/> | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://repo.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz> | Python source package download (3.11.15 in the versioned images, ${PYTHON_VERSION} in the generic Dockerfile). |
| Introduced by open source | N/A | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://www.hiascend.com/> | Public Huawei website used as the Referer header when downloading CANN packages. |
| Introduced by open source | <https://github.com/llvm/llvm-project> | docker/3.2.1-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://github.com/llvm/llvm-project.git> | LLVM source code repository. |
| Introduced by open source | <https://gitee.com/mirrors/llvm-project> | docker/3.2.2-*/Dockerfile | <https://gitee.com/mirrors/llvm-project.git> | LLVM source code mirror used by the 3.2.2 images. |
| Introduced by open source | <https://hub.docker.com> | docker/3.2.2-*/Dockerfile | <https://docker.m.daocloud.io> | DaoCloud Docker Hub mirror used to pull the Ubuntu/openEuler base images. |
| Introduced by open source | <https://mirrors.cloud.tencent.com> | docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://mirrors.cloud.tencent.com/pytorch-wheels/whl/cpu/> | Tencent PyTorch wheel mirror used to install the torch CPU package. |
| Introduced by open source | <https://ubuntu.com> | docker/Dockerfile | <http://archive.ubuntu.com> | Ubuntu apt source replaced by the Huawei mirror. |
| Introduced by open source | <https://mirrors.huaweicloud.com> | docker/Dockerfile | <http://mirrors.huaweicloud.com> | Huawei mirror replacing the Ubuntu apt sources. |
| Introduced by open source | <https://ubuntu.com> | docker/Dockerfile | <http://security.ubuntu.com> | Ubuntu security apt source replaced by the Huawei mirror. |
| Introduced by open source | <https://repo.huaweicloud.com> | docker/Dockerfile | <https://repo.huaweicloud.com/repository/pypi/simple> | Python pip source configured in the image. |
| Introduced by open source | <https://github.com/pytorch/pytorch> | docker/Dockerfile & docker/3.2.1-*/Dockerfile | <https://download.pytorch.org/whl/cpu> | PyTorch CPU wheel index. |
| Self-developed | N/A | docker/Dockerfile | <https://swr.cn-southwest-2.myhuaweicloud.com> | Default CANN base image (CANN_BASE_IMAGE) pulled from Huawei Cloud SWR. |
| Self-developed | N/A | third_party/ascend/build/setup_patch.py | <https://gitcode.com/Ascend/Triton-distributed-ascend.git> | Triton-distributed submodule repository. |
| Self-developed | N/A | third_party/ascend/build/setup_patch.py | <https://github.com/triton-lang/triton-ascend/> | Triton-Ascend project URL injected into package metadata. |
| Self-developed | N/A | third_party/ascend/build/setup_patch.py & third_party/ascend/Triton-distributed-ascend/python/setup.py | `https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz` | Download address of the prepared LLVM tool. |
| Introduced by open source | <https://llvm.org/LICENSE.txt> | third_party/ascend/include/Dialect/TritonAscend/IR/ & third_party/ascend/lib/Dialect/TritonAscend/IR/ | <https://llvm.org/LICENSE.txt> | Apache License notice in source files. |
| Introduced by open source | <https://netlib.org/cephes/> | third_party/ascend/language/cann/libdevice.py | <https://netlib.org/cephes/> | Function source statement. |
| Introduced by open source | <https://github.com/d3/d3> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://d3js.org/d3.v7.min.js> | D3.js library loaded by the exported CFG HTML page. |
| Introduced by open source | <https://github.com/dagrejs/dagre-d3> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://unpkg.com/dagre-d3@0.6.4/dist/dagre-d3.min.js> | dagre-d3 rendering library loaded by the exported CFG HTML page. |
| Introduced by open source | <https://github.com/dagrejs/dagre> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://unpkg.com/dagre@0.8.5/dist/dagre.min.js> | dagre layout library loaded by the exported CFG HTML page. |
