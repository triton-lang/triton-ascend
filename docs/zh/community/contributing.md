# 贡献指南

感谢你对 Triton-Ascend 项目的关注！本文档将帮助你了解如何为项目做出贡献。

## 开发者来源认证（DCO）

所有提交需包含 `Signed-off-by:` 行，使用 `git commit -s` 自动添加：

```bash
git commit -s -m "your commit message"
```

这会在提交信息末尾自动添加一行 `Signed-off-by: Your Name <your.email@example.com>`，表明你确认该贡献的来源和授权。

## 开发环境搭建

### 不依赖 NPU 的本地环境（代码规范检查）

Triton-Ascend 的构建依赖 `torch_npu`，仅支持 Linux。但代码规范检查和基础测试可以在任意系统上进行：

```bash
git clone https://github.com/{your_username}/triton-ascend.git
cd triton-ascend

pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### 依赖 NPU 的完整开发环境

请先按照 [安装指南](installation_guide.md) 完成 Python、CANN 和 torch_npu 的安装配置。

**快速安装**：

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
# Clone AscendNPU-IR
git submodule update --init --depth 1

# Optional: build with local LLVM
# export LLVM_SYSPATH=/path/to/LLVM

pip install -e .
```

**手动安装**：

前置条件：安装好LLVM 22版本，安装方法请参考：https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/installation_guide.md#%E6%89%8B%E5%8A%A8%E5%AE%89%E8%A3%85---%E5%9F%BA%E4%BA%8Ellvm%E6%9E%84%E5%BB%BA，LLVM_INSTALL_PREFIX为LLVM的安装路径

```bash
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

**Docker 安装**：

```bash
docker build \
  --build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
  -t triton-ascend-image:latest -f ./docker/Dockerfile .
```

### 安装开发依赖

```bash
pip install -r python/requirements.txt
pip install -r python/test-requirements.txt
```

## 代码风格

### 编码规范

- **Python**：遵循 [PEP 8](https://pep8.org/) 编码风格
- **C++**：遵循 [LLVM 编码规范](https://llvm.org/docs/CodingStandards.html)

### 代码检查工具

项目使用 [pre-commit](https://pre-commit.com/) 管理代码检查，包含以下工具：

| 工具 | 用途 |
|---|---|
| ruff | Python 代码检查与格式化（行宽 120） |
| yapf | Python 代码格式化（基于 PEP 8，行宽 120） |
| clang-format | C/C++ 代码格式化 |
| mypy | Python 类型检查 |
| pre-commit hooks | 尾部空格、文件末尾换行、YAML/TOML 检查、大文件检测、私钥检测等 |

### 安装并运行 pre-commit

```bash
pip install pre-commit
pre-commit install        # Install git hook, runs automatically on commit
pre-commit run --all-files  # Manually check all files
```

提交 PR 前建议运行：

```bash
pre-commit run --from-ref origin/main --to-ref HEAD
```

### 单元测试规范

- **Python 测试**：使用 [pytest](https://docs.pytest.org/)
- **C++ 测试**：使用 [Google Test](https://github.com/google/googletest/blob/master/docs/primer.md)
- 测试用例的设计意图应通过命名清晰反映，测试用例的设计请参考https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_gather.py

## 运行测试

### 测试目录结构

| 目录 | 内容 |
|---|---|
| `third_party/ascend/unittest/pytest_ut/` | Ascend 特有算子单元测试 |
| `third_party/ascend/unittest/autotune_ut/` | Ascend autotune 测试 |
| `third_party/ascend/unittest/kernels/` | 第三方算子库验证（vLLM 等） |

### 使用 pytest

```bash

# Run Ascend-specific operator unit tests
python -m pytest third_party/ascend/unittest/pytest_ut/ -v -n 8

# Run a single specified test file
python -m pytest third_party/ascend/unittest/pytest_ut/test_index_select_inductor.py -v
```

### PyTorch Inductor 验证

Triton-Ascend 支持 PyTorch Inductor 后端，可通过以下方式验证 Inductor 集成：

```bash

# Run Inductor operator tests
python -m pytest third_party/ascend/unittest/pytest_ut/test_index_select_inductor.py -v
```

也可以编写使用 `torch.compile(..., backend="inductor")` 的测试来验证 Inductor 集成，参考 `third_party/ascend/tutorials/07-profiler.py` 中的 `test_inductor_add` 示例。

### 第三方算子库验证

Triton-Ascend 提供了第三方算子库的集成验证框架，用于确保基于 Triton 的算子库在 Ascend NPU 上的正确性。

**vLLM Kernels 验证**：

```bash
# Run all kernel verifications
python -m pytest third_party/ascend/unittest/kernels/test_triton_kernel.py -v

# Run specified kernel verification
python -m pytest third_party/ascend/unittest/kernels/test_triton_kernel.py -v --kernel={kernel_name}
```

当前已覆盖的 vLLM kernel 包括：attention、rope、layer_norm、fused_gdn_gating、l2norm 等，完整列表见 `third_party/ascend/unittest/kernels/vllm/` 目录。

**新增 Kernel 测试用例**：

新增第三方 kernel 测试用例的流程：
1. 在 GPU 上准备 golden 数据（输入和期望输出），保存为 `.pt` 文件
2. 在 `third_party/ascend/unittest/kernels/{library}/` 下新增 kernel 算子文件
3. 将 `.pt` 文件上传至 OBS 桶：`https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/test/kernels/{library}_pt/{kernel_name}.pt`

详细说明参见 `third_party/ascend/unittest/kernels/README.md`。

## Fork-Pull 开发模式

### 1. Fork 仓库

在 GitHub 上 Fork [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) 到自己的账号下。

### 2. 克隆并配置远程仓库

```bash
git clone https://github.com/{your_username}/triton-ascend.git
cd triton-ascend
git submodule update --init --depth 1
git remote add upstream https://github.com/triton-lang/triton-ascend.git
```

### 3. 创建开发分支

```bash
git checkout -b {feature_branch_name} origin/main
git fetch upstream
git rebase upstream/main
```

### 4. 开发与自测

开发完成后，请确保：
- 代码通过 pre-commit 检查
- 新增代码有对应的测试用例
- 所有相关测试通过
- 如修改了算子实现，建议运行 Inductor 验证和 kernel 对比测试

### 5. 提交代码

```bash
git add {changed_files}
git commit -s -m "简要描述本次修改"
git push origin {feature_branch_name}
```

提交信息请遵循 [How to Write a Git Commit Message](https://cbea.ms/git-commit/#why-not-how) 规范。

### 6. 创建 Pull Request

在 GitHub 上从你的 fork 仓库开发分支向 `triton-lang/triton-ascend` 的 `main` 分支创建 Pull Request。CI 流水线将自动运行。

## PR 规范

### PR 标题前缀

PR 标题建议使用以下前缀标明分类：

| 前缀 | 说明 |
|---|---|
| `[BugFix]` | Bug 修复 |
| `[Kernel]` | 算子相关 |
| `[Core]` | 核心模块 |
| `[Feature]` | 新增特性 |
| `[Refactor]` | 代码重构 |
| `[Revert]` | 代码回滚 |
| `[Perf]` | 性能优化 |
| `[Doc]` | 文档更新 |
| `[Test]` | 测试相关 |
| `[CI]` | CI/CD 相关 |
| `[Misc]` | 其他杂项 |

### PR 检查清单

- [ ] 非琐碎修改（如非拼写修正）
- [ ] 遵循 [提交信息规范](https://cbea.ms/git-commit/#why-not-how)
- [ ] 已运行 `pre-commit run --from-ref origin/main --to-ref HEAD`
- [ ] 已添加测试用例，或说明无需测试的原因
- [ ] 如添加 lit 测试，遵循 [MLIR 测试最佳实践](https://mlir.llvm.org/getting_started/TestingGuide/#filecheck-best-practices)

### PR 审核与合并

- PR 需获得至少 2 个 LGTM（Looks Good To Me）和 1 个 Approval 后方可合并
- 审核人不得在自己的 PR 上添加 LGTM
- 创建 PR 后请尽快合并，以降低合并冲突风险

## Issue 规范

### 提交 Issue

报告问题时，请包含以下信息：

- 软件版本（Triton-Ascend、Python、OS 等）
- 问题类型（Bug 报告还是功能请求）
- 添加对应标签
- 问题描述：发生了什么？
- 预期行为：应该发生什么？
- 复现步骤：尽可能详细

### 问题协作

- 如发现未解决的 Issue 正是你打算处理的，请先在 Issue 下评论说明
- 对于打开较久的 Issue，处理前请先确认问题是否仍然存在
- 如自行解决了自己报告的 Issue，关闭前请通知其他人

## 注意事项

- 避免提交不相关的变更
- 保持提交历史简洁有序
- 创建 PR 前请 rebase 上游最新代码
- 对于 Bug 修复 PR，请链接所有相关 Issue

## 更多信息

- [安全须知](../../SECURITYNOTE_zh.md)
- [治理文档](governance.md)
- [常见问题](../FAQ.md)
