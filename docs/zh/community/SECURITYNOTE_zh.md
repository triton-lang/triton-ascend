# 安全声明

本文档介绍Triton-Ascend的安全声明，涵盖系统安全加固、运行用户建议、文件权限控制、构建安全声明与公网地址声明，适用于安装和使用Triton-Ascend的用户。

## 系统安全加固

建议用户在系统中配置开启ASLR（Address Space Layout Randomization，地址空间布局随机化），推荐使用级别2（完全随机化模式），可参考以下方式进行配置：

```bash
echo 2 > /proc/sys/kernel/randomize_va_space
```

## 运行用户建议

从安全性及权限最小化角度考虑，不建议通过root等管理员类型账户使用Triton-Ascend。

## 文件权限控制

1. 建议用户对个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，设定的权限建议参考[文件权限参考](#file-permission-reference)进行设置。

2. 用户在安装和使用过程中需要做好权限控制，建议按[文件权限参考](#file-permission-reference)进行设置。

<a id="file-permission-reference"></a>

### 文件权限参考

|   类型                             |   Linux权限参考最大值   |
|----------------------------------- |-----------------------|
|  用户主目录                         |   750（rwxr-x---）     |
|  程序文件（含脚本文件、库文件等）     |   550（r-xr-x---）     |
|  程序文件目录                       |   550（r-xr-x---）     |
|  配置文件                           |   640（rw-r-----）     |
|  配置文件目录                       |   750（rwxr-x---）     |
|  日志文件（记录完毕或者已经归档）     |   440（r--r-----）     |
|  日志文件（正在记录）                |   640（rw-r-----）    |
|  日志文件目录                       |   750（rwxr-x---）     |
|  Debug文件                         |   640（rw-r-----）      |
|  Debug文件目录                      |   750（rwxr-x---）     |
|  临时文件目录                       |   750（rwxr-x---）     |
|  维护升级文件目录                   |   770（rwxrwx---）      |
|  业务数据文件                       |   640（rw-r-----）      |
|  业务数据文件目录                   |   750（rwxr-x---）      |
|  密钥组件、私钥、证书、密文文件目录   |   700（rwx------）      |
|  密钥组件、私钥、证书、密文文件       |   600（rw-------）     |
|  加解密接口、加解密脚本              |   500（r-x------）      |

## 构建安全声明

Triton-Ascend支持源码编译安装，在编译时会下载依赖的第三方库并执行构建shell脚本，在编译过程中会产生临时程序文件和编译目录。用户可根据需要自行对源代码目录内的文件进行权限管控，降低安全风险。

## 公网地址声明

在Triton-Ascend的配置文件和脚本中存在[公网IP地址](#public-network-addresses)。

<a id="public-network-addresses"></a>

### 公网地址

| 类型     | 开源代码地址                                                                                     | 文件名                                      | 公网IP地址/公网URL地址/域名/邮箱地址                                                                 | 用途说明                          |
|----------|------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| 自主创新 | 不涉及 | .github/workflows/{DynamicCVPipeline-ci,ascend-documentation,integration-tests-ascend,sync-upstream,tla-tests,upload-wheels}.yml | <https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com> | 用于下载CANN软件包和CI构建产物的OBS桶。 |
| 开源引入 | <https://repo.huaweicloud.com> | .github/workflows/{ascend-documentation,ascend-wheels,integration-tests-ascend,sync-upstream,tla-tests}.yml | <https://repo.huaweicloud.com/ascend/repos/pypi> | CI安装依赖使用的Ascend PyPI源。 |
| 开源引入 | 不涉及 | .github/workflows/{ascend-llvm-build,ascend-wheels,npuir-build,upload-wheels}.yml | <https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_${ARCH}.tar.gz> | obsutil客户端下载地址（x86_64工作流使用obsutil_linux_amd64.tar.gz）。 |
| 自主创新 | 不涉及 | .github/workflows/{ascend-llvm-build,npuir-build,upload-wheels}.yml | <https://obs.cn-southwest-2.myhuaweicloud.com> | obsutil配置的cn-southwest-2区域OBS端点。 |
| 自主创新 | 不涉及 | .github/workflows/{ascend-wheels,upload-wheels}.yml | <https://obs.cn-north-4.myhuaweicloud.com> | obsutil配置的OBS端点，用于上传和下载wheel制品。 |
| 开源引入 | 不涉及 | .github/workflows/doc-review.yml | <https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1> | 文档评审工作流调用deepseek-v4-flash模型的火山引擎API网关地址（可通过DOC_REVIEW_LLM_API_BASE覆盖）。 |
| 开源引入 | <https://github.com/triton-lang/triton.git> | .github/workflows/sync-upstream.yml | <https://github.com/triton-lang/triton.git> | 源码同步工作流使用的Triton上游仓库。 |
| 开源引入 | <https://gitcode.com/Ascend/AscendNPU-IR.git> | .gitmodules | <https://gitcode.com/Ascend/AscendNPU-IR.git> | AscendNPU IR源码仓地址。 |
| 自主创新 | 不涉及 | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/> | CANN工具包与算子包：Ascend-cann-{toolkit,910b-ops,950-ops,A3-ops}_{version}_linux-${ARCH}.run。 |
| 开源引入 | <https://mirrors.ustc.edu.cn> | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://mirrors.ustc.edu.cn/pypi/simple> | 默认Python pip源（PIP_SOURCE）。 |
| 开源引入 | <https://www.python.org/> | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://repo.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz> | Python源码包下载（版本化镜像中为3.11.15，通用Dockerfile使用${PYTHON_VERSION}）。 |
| 开源引入 | 不涉及 | docker/3.2.1-*/Dockerfile & docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://www.hiascend.com/> | 下载CANN软件包时用作Referer的华为公网站点。 |
| 开源引入 | <https://github.com/llvm/llvm-project> | docker/3.2.1-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://github.com/llvm/llvm-project.git> | 依赖的LLVM源码仓。 |
| 开源引入 | <https://gitee.com/mirrors/llvm-project> | docker/3.2.2-*/Dockerfile | <https://gitee.com/mirrors/llvm-project.git> | 3.2.2镜像使用的LLVM源码镜像。 |
| 开源引入 | <https://hub.docker.com> | docker/3.2.2-*/Dockerfile | <https://docker.m.daocloud.io> | 用于拉取Ubuntu/openEuler基础镜像的DaoCloud Docker Hub镜像站。 |
| 开源引入 | <https://mirrors.cloud.tencent.com> | docker/3.2.2-*/Dockerfile & docker/triton_ascend-cann-torch_npu.Dockerfile | <https://mirrors.cloud.tencent.com/pytorch-wheels/whl/cpu/> | 用于安装torch CPU包的腾讯PyTorch wheel镜像。 |
| 开源引入 | <https://ubuntu.com> | docker/Dockerfile | <http://archive.ubuntu.com> | 被华为镜像替换的Ubuntu apt源。 |
| 开源引入 | <https://mirrors.huaweicloud.com> | docker/Dockerfile | <http://mirrors.huaweicloud.com> | 替换Ubuntu apt源的华为镜像站。 |
| 开源引入 | <https://ubuntu.com> | docker/Dockerfile | <http://security.ubuntu.com> | 被华为镜像替换的Ubuntu安全apt源。 |
| 开源引入 | <https://repo.huaweicloud.com> | docker/Dockerfile | <https://repo.huaweicloud.com/repository/pypi/simple> | 镜像中配置的Python pip源。 |
| 开源引入 | <https://hub.docker.com> | docker/Dockerfile & docker/3.2.1-*/Dockerfile | <docker.io> | Docker Hub基础镜像（ubuntu:22.04、library/debian:12、openeuler/openeuler:24.03）。 |
| 开源引入 | <https://github.com/pytorch/pytorch> | docker/Dockerfile & docker/3.2.1-*/Dockerfile | <https://download.pytorch.org/whl/cpu> | PyTorch CPU wheel索引。 |
| 自主创新 | 不涉及 | third_party/ascend/build/setup_patch.py | <https://gitcode.com/Ascend/Triton-distributed-ascend.git> | Triton-distributed子模块仓库。 |
| 自主创新 | 不涉及 | third_party/ascend/build/setup_patch.py | <https://github.com/triton-lang/triton-ascend/> | 写入包元数据的Triton-Ascend项目地址。 |
| 自主创新 | 不涉及 | third_party/ascend/build/setup_patch.py & third_party/ascend/Triton-distributed-ascend/python/setup.py | `https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz` | 用于下载预编译的LLVM工具。 |
| 开源引入 | <https://llvm.org/LICENSE.txt> | third_party/ascend/include/Dialect/TritonAscend/IR/ & third_party/ascend/lib/Dialect/TritonAscend/IR/ | <https://llvm.org/LICENSE.txt> | 源码中的Apache许可证声明。 |
| 开源引入 | <https://netlib.org/cephes/> | third_party/ascend/language/cann/libdevice.py | <https://netlib.org/cephes/> | 函数来源声明。 |
| 开源引入 | <https://github.com/d3/d3> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://d3js.org/d3.v7.min.js> | 导出的CFG HTML页面加载的D3.js库。 |
| 开源引入 | <https://github.com/dagrejs/dagre-d3> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://unpkg.com/dagre-d3@0.6.4/dist/dagre-d3.min.js> | 导出的CFG HTML页面加载的dagre-d3渲染库。 |
| 开源引入 | <https://github.com/dagrejs/dagre> | third_party/ascend/lib/TritonToGraph/ControlFlowGraph.cpp | <https://unpkg.com/dagre@0.8.5/dist/dagre.min.js> | 导出的CFG HTML页面加载的dagre布局库。 |
