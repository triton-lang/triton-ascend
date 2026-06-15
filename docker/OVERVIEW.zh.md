# Triton编译框架 Triton-Ascend

> [English](./OVERVIEW.md) | [中文]

## 快速参考

- Triton-Ascend 由[triton-ascend 代码仓](https://github.com/triton-lang/triton-ascend/)维护
- 从哪里获取帮助
    - [triton-Ascend 代码仓](https://github.com/triton-lang/triton-ascend/)
    - [Triton-Ascend 资料站](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
    - [问题反馈](https://github.com/triton-lang/triton-ascend/issues)

--- 

# Triton-Ascend
Triton-Ascend是面向昇腾平台构建的Triton编译框架，旨在让Triton代码能够在昇腾硬件上高效运行。详见[Triton-Ascend](https://github.com/triton-lang/triton-ascend/blob/main/README_zh.md)。
# 支持的Tags及Dockerfile链接
## Tag 规范
Tag遵循以下格式：<br/>
`<triton-ascend版本>-<芯片系列>-<操作系统>-<python版本>`

| 字段              | 示例值                        | 说明              |
|-------------------|-------------------------------|-------------------|
| triton-ascend版本 | 3.2.1                         | triton-ascend版本 |
| 芯片系列          | 910b、a3、950                 | 目标昇腾芯片系列  |
| 操作系统          | ubuntu22.04、openeuler24.03   | 基础操作系统      |
| python版本        | py3.11                        | Python版本        |

## Triton-Ascend 镜像
### Release 3.2.1
#### 镜像内关键组件
| 组件              | 版本          | 
|-----------------|-------------|
| Triton-Ascend   | 3.2.1       | 
| CANN            | 9.0.0       |
| Torch-npu       | 2.7.1.post4 |

#### 镜像列表 

| 镜像标签                            | Dockerfile         | 镜像下载命令                                                       |
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


# 快速开始
## 运行Triton-Ascend容器
```
# 假设您的NPU设备型号是A3,且设备安装在/dev/davinci1上，并且您的NPU驱动程序安装在/usr/local/Ascend上：
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
## 如何本地构建
**arm64架构**
```
docker build \
--network host \
--build-arg TARGETPLATFORM=linux/arm64 \
-t triton:3.2.1-a3-ubuntun22.04-py3.11-aarch64 \
-f Dockerfile .
```
**x86_64架构**
```
docker build \
--network host \
--build-arg TARGETPLATFORM=linux/amd64 \
-t triton:3.2.1-a3-ubuntun22.04-py3.11-x86_64 \
-f Dockerfile .
```

## 如何二次开发
```
# 以triton-ascend镜像为基础镜像，叠加用户软件
FROM quay.io/ascend/triton:3.2.1-a3-ubuntu22.04-py3.11
RUN apt update -y && \
    apt install wget \
    ...
```
# 支持的硬件
| 芯片系列  | 产品示例                        | 架构          |
|-----------|---------------------------------|---------------|
| 昇腾910b  | Atlas 800T A2、Atlas 900 A2 PoD | ARM64、x86_64 |
| 昇腾A3    | Atlas 800T A3                   | ARM64、x86_64 |
| 昇腾950   | 950PR系列                       | ARM64、x86_64 |

# 许可证
查看镜像中包含的CANN、Torch-npu、Triton-Ascend软件的[许可证信息](https://www.hiascend.com/zh/software/protocol)。<br/>
与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
