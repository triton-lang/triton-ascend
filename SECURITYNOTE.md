# Triton-Ascend Security Note

## System Security Hardening

You are advised to enable the address space layout randomization (ASLR) (level 2) in the system. You can perform the following operation to enable it:

    echo 2 > /proc/sys/kernel/randomize_va_space

## Suggestions on Running Users

To ensure security and minimize permissions, you are advised not to use administrator accounts such as **root**.

## File Permission Control

1. You are advised to take security measures such as permission control on sensitive files, such as personal privacy data and business assets. For details about how to set the permissions, see the "File Permission Reference" section.

2. During the installation and use, you are advised to control the permissions. For details about how to set the permissions, see [File Permission Reference](#file-permission-reference).


##### File permission reference

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
|  Service data file directory                  |   750 (rwxr-x---)     |
|  Key component, private key, certificate, and ciphertext file directory  |   700 (rwx------)      |
|  Key components, private keys, certificates, and ciphertext files      |   600 (rw-------)    |
|  APIs and script files for encryption and decryption             |   500 (r-x------)     |


## Build Security Statement

Triton-Ascend can be installed through source code compilation. During the compilation process, dependent third-party libraries are downloaded and the shell build script is executed. This results in the generation of temporary program files and compilation directories. You can control permissions on files in the source code directory as required to prevent security risks.

## Public IP Address Statement

Public IP addresses are used in the configuration files and scripts of Triton-Ascend. For details, see the "Public IP Addresses" section.

##### Public IP addresses
| Type    | Open-Source Code Address                                                                                    | File Name                                     | Public IP Address/Public URL/Domain Name/Email Address                                                                | Description                         |
|----------|------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| Introduced by open source| https://github.com/triton-lang/triton.git | .gitmodules | https://github.com/triton-lang/triton.git | Address of the Triton source code repository|
| Introduced by open source| https://gitcode.com/Ascend/AscendNPU-IR.git | .gitmodules | https://gitcode.com/Ascend/AscendNPU-IR.git | AscendNPU IR source code repository address|
| Self-developed    | N/A                                                                                        | docker/devdocker/setup_triton-ascend_dev.sh | https://gitcode.com/Ascend/triton-ascend.git                                                          | Address of the Triton-Ascend source code repository                |
| Self-developed    | N/A                                                                                        | ascend/examples/generalization_cases/run_daily.sh & scripts/prepare_build.sh | https://gitee.com/shijingchang/triton.git                                                           | Build dependency code repository                |
| Self-developed    | N/A                                                                                        | setup.py                                   | https://gitcode.com/Ascend/triton-ascend/                                                             | Address of the Triton-Ascend source code repository|
| Introduced by open source| https://gitclone.com                                                            | scripts/prepare_build.sh                   | https://gitclone.com/github.com/llvm/llvm-project.git                                               | LLVM source code repository   |
| Introduced by open source| https://repo.huaweicloud.com                                            | scripts/prepare_build.sh                           | https://repo.huaweicloud.com/repository/pypi/simple                                                | Used to configure the pybind11 download link.|
| Introduced by open source| https://pypi.tuna.tsinghua.edu.cn                                                                                         | docker/devdocker/triton-ascend_dev.dockerfile | https://pypi.tuna.tsinghua.edu.cn/simple                                                             | Python pip source configuration        |
| Introduced by open source| https://triton-ascend-artifacts.obs.myhuaweicloud.com | setup.py |https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz | Used to download the prepared LLVM tool.|
| Introduced by open source| https://bootstrap.pypa.io/get-pip.py | docker/develop_env.dockerfile |https://bootstrap.pypa.io/get-pip.py | Used to automatically install pip.|
| Introduced by open source | https://llvm.org/LICENSE.txt | third_party/ascend/include/Dialect/TritonAscend/IR/* & third_party/ascend/lib/Dialect/TritonAscend/IR/* | https://llvm.org/LICENSE.txt | Apache License |
| Introduced by open source | https://netlib.org/cephes/ | third_party/ascend/language/cann/libdevice.py | https://netlib.org/cephes/ | Function source statement. |