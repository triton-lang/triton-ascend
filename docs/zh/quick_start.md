# 快速入门

## 项目简介

Triton-Ascend 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，提供高效的核函数自动调优、算子编译及部署能力，支持 Ascend Atlas A2/A3 等系列产品，
兼容 Triton 核心语法的同时，针对昇腾 NPU 特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

## 在线文档
我们提供了完整的在线文档与网络资料，涵盖环境搭建、算子开发、调优实践以及常见问题说明，方便用户快速上手与深入使用，详情请参考 [在线文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)

## 环境要求
### 硬件要求
支持的操作系统: linux(aarch64/x86_64)

支持的 Ascend 产品: Atlas A2/A3 系列

最小硬件配置: 单卡 32GB 显存（推荐）

### 软件依赖
Python(**py3.9-py3.11**)，CANN_TOOLKIT，CANN_OPS，以及[requirements.txt](../../requirements.txt)和[requirements_dev.txt](../../requirements_dev.txt)等。

您可以访问昇腾社区官网，根据其提供的[社区软件安装指引](https://www.hiascend.com/cann/download)完成 CANN 的安装与配置。开发者选择CANN版本、产品系列、CPU架构、操作系统和安装方式便可找到对应的安装命令。


在安装过程中，CANN 版本“**{version}**”请选择如下版本之一。建议下载安装 8.5.0 版本:
- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，`${HOME}`为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将```source ${HOME}/Ascend/ascend-toolkit/set_env.sh```命令写入环境变量配置文件（如.bashrc文件）。

需要根据您实际使用的昇腾卡型号选择对应的配套版本(建议8.5.0版本)，并且安装CANN的时间大概在5-10分钟，请耐心等待安装完成。

requirements的安装可以参考如下：
```shell
pip install -r requirements.txt -r requirements_dev.txt
```

## 环境搭建
用户可根据[安装指南](installation_guide.md)的环境准备章节步骤搭建Triton-Ascend环境。

### Triton-Ascend 软件包获取
用户可以直接命令行安装最新的稳定版本包。
```shell
pip install triton-ascend
```

- 注意：社区 Triton 和 Triton-Ascend 不能同时存在。在安装依赖Triton的其他软件时，会自动安装社区 Triton，将覆盖掉已安装的 Triton-Ascend 目录。
此时也需要先卸载社区 Triton，再安装 Triton-Ascend。

也可以在 [下载地址](https://test.pypi.org/project/triton-ascend/#history) 中自行选择nightly包进行下载然后本地安装。

- 注意1：如果您选择自行下载nightly包安装，请在选择Triton-Ascend包时选择对应您服务器的python版本以及架构(aarch64/x86_64)。
- 注意2：nightly是每日构建的包，开发者提交mr频繁，没有经过稳定的测试，可能存在功能上的bug，请知悉。

## 快速使用Docker 安装环境
我们提供了Dockerfile帮助您安装Docker环境镜像。安装过程将会自动从CANN官网中下载安装对应的CANN Toolkit和Kernel包，需要您通过`--build-arg`指定您机器需要安装的CANN相关参数。

| 参数名称 | 默认值 | 可选值                                   |
| -------- | ------ |---------------------------------------|
| CHIP_TYPE | A3     | A3、910b                               |
| CANN_VERSION | 8.5.0（推荐） | 8.5.0、8.3.RC1、8.3.RC2、8.2.RC1、8.2.RC2 |

您可以通过 npu-smi 命令查看系统上的NPU型号。

不同`CHIP_TYPE`选项对应的机器可参考：

| 选项序号 | **CHIP_TYPE 参数值** | 对应机器/产品系列 |                 典型整机                 |   别称    |
| :---: |:-----------------:| :---: |:-----------------------------------:|:-------:|
| 1 |       `A3`        | Atlas A3 训练系列产品 |        Atlas 900 A3 SuperPoD        |  910C   |
| 2 |      `910b`       | Atlas A2 训练系列产品 |            Atlas800T A2             |   A2    |
```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CHIP_TYPE=A3 \
--build-arg CANN_VERSION=8.5.0 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```
根据该镜像启动容器，可以参考下面的命令：
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
-v /usr/local/Ascend:/usr/local/Ascend \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
triton-ascend-image:latest \
/bin/bash

# 进入容器
docker exec -u root -it triton-ascend_container /bin/bash
```

## 运行Triton示例

运行实例: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 运行tutorials示例：
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。
```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```
