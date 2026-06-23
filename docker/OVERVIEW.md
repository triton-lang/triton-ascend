# Triton compilation framework Triton-Ascend

> [English] | [中文](./OVERVIEW.zh.md)

## Quick Reference

- Triton-Ascend is maintained by[triton-ascend Code Repository](https://github.com/triton-lang/triton-ascend/)
- Where to get help
    - [triton-Ascend Code Repository](https://github.com/triton-lang/triton-ascend/)
    - [Triton-Ascend Data Website](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
    - [Issue Tracker](https://github.com/triton-lang/triton-ascend/issues)

---

# Triton-Ascend

Triton-Ascend is a Triton compilation framework built for the Ascend platform, aiming to enable Triton code to run efficiently on Ascend hardware. For details, see [Triton-Ascend](https://github.com/triton-lang/triton-ascend/blob/main/README.md).

# Supported Tags and Dockerfile

## Tag specifications

Tag follow the following format：<br/>
`<triton-ascend version>-<chip series>-<OS>-<python version>`

| Field                 | Example Value               | Description               |
|-----------------------|-----------------------------|---------------------------|
| triton-ascend version | 3.2.1                       | triton-ascend version     |
| chip series           | 910b, a3, 950               | Target Ascend chip series |
| OS                    | ubuntu22.04, openeuler24.03 | Basic OS                  |
| python version        | py3.11                      | Python version            |

## Triton-Ascend Images

### Release 3.2.1

#### Key Image Components

| Components      | Version     |
|-----------------|-------------|
| Triton-Ascend   | 3.2.1       |
| CANN            | 9.0.0       |
| Torch-npu       | 2.7.1.post4 |

#### Image List

| Tag                              | Dockerfile        | Docker Pull Command                                                |
|----------------------------------|-------------------|--------------------------------------------------------------------|
| 3.2.1-910b-debian12-py3.11       | [Dockerfile](3.2.1-910b-debian12-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-910b-debian12-py3.11       |
| 3.2.1-910b-ubuntu22.04-py3.11    | [Dockerfile](3.2.1-910b-ubuntu22.04-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-910b-ubuntu22.04-py3.11    |
| 3.2.1-910b-openeuler24.03-py3.11 | [Dockerfile](3.2.1-910b-openeuler24.03-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-910b-openeuler24.03-py3.11 |
| 3.2.1-a3-debian12-py3.11         | [Dockerfile](3.2.1-a3-debian12-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-a3-debian12-py3.11         |
| 3.2.1-a3-ubuntu22.04-py3.11      | [Dockerfile](3.2.1-a3-ubuntu22.04-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-a3-ubuntu22.04-py3.11      |
| 3.2.1-a3-openeuler24.03-py3.11   | [Dockerfile](3.2.1-a3-openeuler24.03-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-a3-openeuler24.03-py3.11   |
| 3.2.1-950-debian12-py3.11        | [Dockerfile](3.2.1-950-debian12-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-950-debian12-py3.11        |
| 3.2.1-950-ubuntu22.04-py3.11     | [Dockerfile](3.2.1-950-ubuntu22.04-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-950-ubuntu22.04-py3.11     |
| 3.2.1-950-openeuler24.03-py3.11  | [Dockerfile](3.2.1-950-openeuler24.03-py3.11/Dockerfile) | docker pull quay.io/ascend/triton:3.2.1-950-openeuler24.03-py3.11  |

# Quick Start

## Running the Triton-Ascend Container

```bash
# Assume that your NPU device model is A3, the device is installed in /dev/davinci1, and the NPU driver is installed in /usr/local/Ascend:
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
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
quay.io/ascend/triton:3.2.1-a3-ubuntu22.04-py3.11 \
/bin/bash

```

## How to Build Locally?

**arm64**

```bash
docker build \
--network host \
--build-arg TARGETPLATFORM=linux/arm64 \
-t triton:3.2.1-a3-ubuntun22.04-py3.11-aarch64 \
-f Dockerfile .
```

**amd64**

```bash
docker build \
--network host \
--build-arg TARGETPLATFORM=linux/amd64 \
-t triton:3.2.1-a3-ubuntun22.04-py3.11-x86_64 \
-f Dockerfile .
```

## How to Perform Secondary Development?

```bash
# Using the triton-ascend image as the base image and adding user software to it.
FROM quay.io/ascend/triton:3.2.1-a3-ubuntu22.04-py3.11
RUN apt update -y && \
    apt install wget \
    ...
```

# Supported hardware

| Chip Series | Product Example                 | Architecture  |
|-------------|---------------------------------|---------------|
| Ascend 910b | Atlas 800T A2, Atlas 900 A2 PoD | arm64, amd64  |
| Ascend A3   | Atlas 800T A3                   | arm64, amd64 |
| Ascend 950  | 950PR Series                    | arm64, amd64 |

# License

View the license information of the CANN, Torch-npu and Triton-Ascend software contained in the image. For details, visit [License](https://www.hiascend.com/en/software/protocol).<br/>
As with all container images, the pre-installed software packages (such as Python and system libraries) may be subject to their own licenses.
