#!/usr/bin/env python3
"""
Fix PO file translations for Triton-Ascend documentation.
This script corrects merged translations, misaligned table data,
and empty msgstr entries across 14 PO files.
"""

import os
import re
import hashlib

BASE_DIR = "/Users/hua/code/triton-ascend/docs/locale/en/LC_MESSAGES"
ZH_DIR = "/Users/hua/code/triton-ascend/docs/zh"

# ============================================================
# Helper: parse PO file into entries
# ============================================================
def parse_po(filepath):
    """Parse PO file into a list of entries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = []
    # Split by blank lines (entry separator)
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        entry = {'header': [], 'msgid': [], 'msgstr': [], 'raw': block}
        state = 'header'

        for line in lines:
            if line.startswith('msgid '):
                state = 'msgid'
                entry['msgid'].append(line)
            elif line.startswith('msgstr '):
                state = 'msgstr'
                entry['msgstr'].append(line)
            elif line.startswith('#') or line.startswith('"'):
                if state == 'header':
                    entry['header'].append(line)
                elif state == 'msgid':
                    entry['msgid'].append(line)
                elif state == 'msgstr':
                    entry['msgstr'].append(line)

        entries.append(entry)

    return entries, content

def write_po(filepath, entries):
    """Write entries back to PO file."""
    lines = []
    for i, entry in enumerate(entries):
        if i > 0:
            lines.append('')
        for h in entry.get('header', []):
            lines.append(h)
        for m in entry.get('msgid', []):
            lines.append(m)
        for s in entry.get('msgstr', []):
            lines.append(s)
    # Ensure trailing newline
    content = '\n'.join(lines)
    if not content.endswith('\n'):
        content += '\n'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def get_entry_text(entry, field='msgid'):
    """Get the full text of a msgid or msgstr field."""
    lines = entry.get(field, [])
    text = ''
    for line in lines:
        if line.startswith(f'{field} '):
            # Handle both msgid "..." and msgid ""
            match = re.match(rf'{field}\s+"(.*)"\s*$', line, re.DOTALL)
            if match:
                text += match.group(1)
        elif line.startswith('"'):
            match = re.match(r'"(.*)"\s*$', line, re.DOTALL)
            if match:
                text += match.group(1)
    return text

def set_entry_text(entry, field, text):
    """Set the text of a msgid or msgstr field, preserving multiline format."""
    # Remove existing lines for this field
    new_lines = []
    for line in entry.get(field, []):
        if line.startswith(f'{field} '):
            new_lines = [line]  # will replace
        elif line.startswith('"'):
            continue  # skip continuation lines
        else:
            new_lines.append(line)

    # Find the first line with the field keyword
    first_line_idx = None
    for j, line in enumerate(entry.get(field, [])):
        if line.startswith(f'{field} '):
            first_line_idx = j
            break

    if first_line_idx is not None:
        # Remove old lines
        old_lines = entry[field][:]
        entry[field] = old_lines[:first_line_idx]

        # Add new formatted text
        if '\n' in text:
            # Multiline text
            first, rest = text.split('\n', 1)
            entry[field].append(f'{field} "{first}"')
            for part in rest.split('\n'):
                entry[field].append(f'"{part}"')
        else:
            entry[field].append(f'{field} "{text}"')

def normalize_msgid(msgid):
    """Remove reference file paths from comments for comparison."""
    return msgid.strip()

# ============================================================
# Helper: translate Chinese to English using known patterns
# ============================================================
TRANSLATION_CACHE = {}

def translate_text(chinese_text, context=''):
    """Translate Chinese text to English.
    Uses a combination of exact match, pattern matching, and manual translation.
    """
    if not chinese_text or chinese_text.strip() == '':
        return chinese_text

    # Check cache
    if chinese_text in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[chinese_text]

    # Direct translation dictionary for common phrases
    translations = {
        # Simple terms
        "贡献指南": "Contributing Guide",
        "开发者来源认证（DCO）": "Developer Certificate of Origin (DCO)",
        "开发环境搭建": "Development Environment Setup",
        "不依赖 NPU 的本地环境（代码规范检查）": "Local Environment Without NPU (Code Style Checks)",
        "依赖 NPU 的完整开发环境": "Full Development Environment with NPU",
        "代码风格": "Code Style",
        "编码规范": "Coding Standards",
        "代码检查工具": "Code Checking Tools",
        "安装并运行 pre-commit": "Install and Run pre-commit",
        "单元测试规范": "Unit Testing Standards",
        "运行测试": "Running Tests",
        "测试目录结构": "Test Directory Structure",
        "使用 pytest": "Using pytest",
        "Fork-Pull 开发模式": "Fork-Pull Development Model",
        "开发目标": "R&D Goals",
        "开发步骤": "Development Procedure",
        "注意事项": "Notes",
        "更多信息": "More Information",
        "工具": "Tool",
        "用途": "Purpose",
        "目录": "Directory",
        "内容": "Content",
        "前缀": "Prefix",
        "说明": "Description",
        "PR 规范": "PR Guidelines",
        "PR 标题前缀": "PR Title Prefixes",
        "PR 检查清单": "PR Checklist",
        "PR 审核与合并": "PR Review and Merge",
        "Issue 规范": "Issue Guidelines",
        "提交 Issue": "Submitting Issues",
        "问题协作": "Issue Collaboration",

        # Governance terms
        "治理": "Governance",
        "使命": "Mission",
        "原则": "Principles",
        "治理机制": "Governance Mechanism",
        "提名与任免": "Nomination and Removal",
        "提名流程": "Nomination Process",
        "定期审视": "Regular Review",
        "维护者与贡献者": "Maintainers and Contributors",
        "姓名": "Name",
        "ID": "ID",
        "加入时间": "Join Date",
        "需要": "No.",
        "贡献者": "Contributor",
        "首次提交日期": "First Contribution Date",
        "PR": "PR",

        # Release policy
        "版本策略": "Release Policy",
        "版本号规则": "Version Numbering",
        "版本分支策略": "Branching Strategy",
        "维护分支与生命周期": "Maintenance Branches and Lifecycle",
        "发布周期": "Release Cycle",
        "发布时间线": "Release Timeline",
        "版本兼容性矩阵": "Version Compatibility Matrix",
        "分支": "Branch",
        "状态": "Status",
        "对应 Triton 版本": "Triton Version",
        "Triton-Ascend 发布版本": "Triton-Ascend Release",
        "维护截止日期": "End of Maintenance",
        "日期": "Date",
        "事件": "Event",

        # Community technical meeting
        "社区技术会议": "Community Technical Meeting",
        "会议内容": "Meeting Content",
        "会议安排": "Meeting Schedule",
        "投票规则": "Voting Rules",
        "决策流程": "Decision-Making Process",
        "常规决策": "Routine Decisions",
        "重大决策": "Major Decisions",

        # Debug guide
        "Triton-Ascend 调试指南": "Triton-Ascend Debugging Guide",
        "1 引言": "1 Overview",
        "编译流程概览": "Compilation Process Overview",
        "临时文件指引": "Temporary File Guide",
        "解释器模式": "Interpreter Mode",
        "调试方法": "Debugging Methods",

        # Architecture difference
        "昇腾与GPU的开发差异": "Development Differences Between Ascend and GPUs",
        "多核任务并行策略": "Multi-Core Task Parallelism Strategy",
        "单核数据搬运策略": "Single-Core Data Transfer Strategy",
        "数据切分Tiling": "Data Tiling",
        "编译优化能力": "Compilation Optimization",
        "AscendNPU IR优化": "Ascend NPU IR Optimization",

        # Migrate from GPU
        "GPU Triton算子迁移": "Migrating Triton Operators from GPUs",
        "通用迁移流程": "General Migration Procedure",
        "迁移 Python 侧设备和运行时接口": "Migrate Python-Side Device and Runtime Interfaces",
        "调整 grid 分核": "Adjust Grid Core Allocation",
        "检查单核数据搬运": "Check Single-Program Data Transfer",
        "检查单核数据运算": "Check Single-Program Computation",
        "迁移示例": "Migration Examples",

        # Performance guidelines
        "NPU高性能编程指南": "NPU High-Performance Programming Guide",
        "合并Grid分核": "Combining Grid Cores",
        "指令并行优化": "Optimizing Instruction Parallelism",
        "数据类型优化": "Optimizing Data Types",

        # Libdevice
        "Libdevice 开发者手册": "Libdevice Developer Guide",
        "SIMT 编译示例": "SIMT Compilation Mode Example",
        "使用 SIMT 编译的 triton kernel 示例": "Triton kernel example with SIMT compilation mode",
        "OP概述": "OP Overview",
        "原型:": "Prototype:",

        # Programming guide headers
        "Triton算子开发指南": "Triton Operator Development Guide",
        "文档组织": "Documentation Structure",
        "通用多核任务并行": "Common Multi-Core Task Parallelism",
        "设置最大硬件核数": "Setting the Maximum Number of Hardware Cores",
        "通用单核数据搬运": "Common Single-Core Data Transfer",
        "设置合适的循环内数据分块大小（BLOCK SIZE）": "Setting the Proper Data Block Size (BLOCK SIZE)",
        "通用单核数据运算": "Common Single-Core Data Computation",
        "代码风格": "Code Style",
        "Tiling优化": "Tiling Optimization",
        "Triton Autotune 自动调优": "Triton Autotune",
        "Vector 算子开发": "Vector Operator Development",
        "Vector 简单算子开发": "Simple Vector Operator Development",
        "Vector 复杂算子开发": "Complex Vector Operator Development",
        "Cube 算子开发": "Cube Operator Development",
        "Cube 简单算子开发": "Simple Cube Operator Development",
        "Cube 复杂算子开发": "Complex Cube Operator Development",
        "CV 融合算子开发": "CV Fusion Operator Development",
        "CV 融合简单算子开发": "Simple CV Fusion Operator Development",
        "CV 融合复杂算子开发": "Complex CV Fusion Operator Development",
        "示例": "Example",
        "示例结构：": "Example structure:",
        "核心结构如下：": "Core structure is as follows:",

        # Profiling
        "Triton-Ascend 性能分析方法": "Triton-Ascend Performance Analysis Method",
        "获取性能数据": "Obtaining Performance Data",
        "上板Profiling": "Board Profiling",
        "算子仿真流水图": "Operator Simulation Pipeline Diagram",
        "分析性能数据": "Analyzing Performance Data",
        "理论参数": "Theoretical Parameters",
        "查找瓶颈": "Locating Bottlenecks",

        # Table headers
        "选项": "Option",
        "能力": "Capability",
        "是否开启": "Enabled or Not",
        "维度": "Dimension",
        "核心结构": "Core Structure",
        "算子类型": "Operator Type",
        "昇腾 NPU (Ascend)": "Ascend NPU",
        "GPU（NVIDIA）": "GPU (NVIDIA)",
        "昇腾（Ascend）": "Ascend",
        "grid 本质": "Essence of grids",
        "核数 / 维度限制": "Limit on the number of cores/dimensions",
        "问题类型": "Issue Type",
        "典型表现/描述": "Typical Symptom/Description",
        "推荐的首要调试方法": "Preferred Debugging Method",
        "章节": "Section",
        "主要内容": "Description",
        "阶段": "Phase",
        "输入": "Input",
        "输出": "Output",
        "工具/组件": "Tool/Component",
        "特性": "Feature",
        "变量": "Variable",
        "作用": "Description",
        "文件类型": "File Type",
        "生成阶段": "Generation Phase",
        "触发条件": "Triggering Condition",
        "清理建议": "Clearance Suggestion",

        # Image alt texts
        "alt text": "alt text",
        "analyse_data_op_summary": "analyse_data_op_summary",
        "analyse_data_waveform": "analyse_data_waveform",
        "analyse_data_code_mapping": "analyse_data_code_mapping",

        # Debug env vars section
        "`TRITON_DEVICE_PRINT=1`": "`TRITON_DEVICE_PRINT=1`",
    }

    if chinese_text in translations:
        result = translations[chinese_text]
        TRANSLATION_CACHE[chinese_text] = result
        return result

    # For text that has no exact match, try to translate using patterns
    # If text contains only ASCII, it's already in English
    if all(ord(c) < 128 for c in chinese_text):
        TRANSLATION_CACHE[chinese_text] = chinese_text
        return chinese_text

    # Log untranslated text
    print(f"  [UNTRANSLATED] '{chinese_text[:80]}...' " if len(chinese_text) > 80 else f"  [UNTRANSLATED] '{chinese_text}'")
    TRANSLATION_CACHE[chinese_text] = chinese_text
    return chinese_text


# ============================================================
# Process community/contributing.po
# ============================================================
def fix_contributing_po():
    filepath = os.path.join(BASE_DIR, 'community', 'contributing.po')
    entries, raw = parse_po(filepath)

    # Read Chinese source for context
    zh_file = os.path.join(ZH_DIR, 'community', 'contributing.md')
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_content = f.read()

    # Build mappings for translated content
    # For list items that are merged, we need to split them

    translations = {
        "贡献指南": "Contributing Guide",
        "感谢你对 Triton-Ascend 项目的关注！本文档将帮助你了解如何为项目做出贡献。": "Thank you for your interest in the Triton-Ascend project! This document will help you understand how to contribute to the project.",
        "开发者来源认证（DCO）": "Developer Certificate of Origin (DCO)",
        "所有提交需包含 `Signed-off-by:` 行，使用 `git commit -s` 自动添加：": "All commits must include a `Signed-off-by:` line. Use `git commit -s` to add it automatically:",
        "这会在提交信息末尾自动添加一行 `Signed-off-by: Your Name <your.email@example.com>`，表明你确认该贡献的来源和授权。": "This will automatically add a line `Signed-off-by: Your Name <your.email@example.com>` at the end of your commit message, indicating that you certify the origin and authorization of the contribution.",
        "开发环境搭建": "Development Environment Setup",
        "不依赖 NPU 的本地环境（代码规范检查）": "Local Environment Without NPU (Code Style Checks)",
        "Triton-Ascend 的构建依赖 `torch_npu`，仅支持 Linux。但代码规范检查和基础测试可以在任意系统上进行：": "Triton-Ascend's build depends on `torch_npu` and only supports Linux. However, code style checks and basic tests can be performed on any system:",
        "依赖 NPU 的完整开发环境": "Full Development Environment with NPU",
        "请先按照 [安装指南](installation_guide.md) 完成 Python、CANN 和 torch_npu 的安装配置。": "Please first follow the [Installation Guide](installation_guide.md) to complete the installation and configuration of Python, CANN, and torch_npu.",
        "**快速安装**：": "**Quick Installation**:",
        "**手动安装**：": "**Manual Installation**:",
        "**Docker 安装**：": "**Docker Installation**:",
        "安装开发依赖": "Install Development Dependencies",
        "代码风格": "Code Style",
        "编码规范": "Coding Standards",
        "**Python**：遵循 [PEP 8](https://pep8.org/) 编码风格": "- **Python**: Follow [PEP 8](https://pep8.org/) coding style",
        "**C++**：遵循 [LLVM 编码规范](https://llvm.org/docs/CodingStandards.html)": "- **C++**: Follow [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)",
        "代码检查工具": "Code Checking Tools",
        "项目使用 [pre-commit](https://pre-commit.com/) 管理代码检查，包含以下工具：": "The project uses [pre-commit](https://pre-commit.com/) to manage code checks, including the following tools:",
        "工具": "Tool",
        "用途": "Purpose",
        "ruff": "ruff",
        "Python 代码检查与格式化（行宽 120）": "Python code linting and formatting (line width 120)",
        "yapf": "yapf",
        "Python 代码格式化（基于 PEP 8，行宽 120）": "Python code formatting (based on PEP 8, line width 120)",
        "clang-format": "clang-format",
        "C/C++ 代码格式化": "C/C++ code formatting",
        "mypy": "mypy",
        "Python 类型检查": "Python type checking",
        "pre-commit hooks": "pre-commit hooks",
        "尾部空格、文件末尾换行、YAML/TOML 检查、大文件检测、私钥检测等": "Trailing whitespace, end-of-file newline, YAML/TOML checks, large file detection, private key detection, etc.",
        "安装并运行 pre-commit": "Install and Run pre-commit",
        "提交 PR 前建议运行：": "Before submitting a PR, it's recommended to run:",
        "单元测试规范": "Unit Testing Standards",
        "**Python 测试**：使用 [pytest](https://docs.pytest.org/)": "- **Python Tests**: Use [pytest](https://docs.pytest.org/)",
        "**C++ 测试**：使用 [Google Test](https://github.com/google/googletest/blob/master/docs/primer.md)": "- **C++ Tests**: Use [Google Test](https://github.com/google/googletest/blob/master/docs/primer.md)",
        "测试用例的设计意图应通过命名清晰反映，测试用例的设计请参考https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_gather.py": "- Test case design intent should be clearly reflected through naming. For test case design, refer to https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_gather.py",
        "运行测试": "Running Tests",
        "测试目录结构": "Test Directory Structure",
        "目录": "Directory",
        "内容": "Content",
        "`third_party/ascend/unittest/pytest_ut/`": "`third_party/ascend/unittest/pytest_ut/`",
        "Ascend 特有算子单元测试": "Ascend-specific operator unit tests",
        "`third_party/ascend/unittest/autotune_ut/`": "`third_party/ascend/unittest/autotune_ut/`",
        "Ascend autotune 测试": "Ascend autotune tests",
        "`third_party/ascend/unittest/kernels/`": "`third_party/ascend/unittest/kernels/`",
        "第三方算子库验证（vLLM 等）": "Third-party operator library validation (vLLM, etc.)",
        "使用 pytest": "Using pytest",
        "PyTorch Inductor 验证": "PyTorch Inductor Validation",
        "Triton-Ascend 支持 PyTorch Inductor 后端，可通过以下方式验证 Inductor 集成：": "Triton-Ascend supports the PyTorch Inductor backend. You can validate Inductor integration through the following methods:",
        "也可以编写使用 `torch.compile(..., backend=\"inductor\")` 的测试来验证 Inductor 集成，参考 `third_party/ascend/tutorials/07-profiler.py` 中的 `test_inductor_add` 示例。": "You can also write tests using `torch.compile(..., backend=\"inductor\")` to validate Inductor integration. Refer to the `test_inductor_add` example in `third_party/ascend/tutorials/07-profiler.py`.",
        "第三方算子库验证": "Third-Party Operator Library Validation",
        "Triton-Ascend 提供了第三方算子库的集成验证框架，用于确保基于 Triton 的算子库在 Ascend NPU 上的正确性。": "Triton-Ascend provides an integration validation framework for third-party operator libraries to ensure the correctness of Triton-based operator libraries on Ascend NPU.",
        "**vLLM Kernels 验证**：": "**vLLM Kernels Validation**:",
        "当前已覆盖的 vLLM kernel 包括：attention、rope、layer_norm、fused_gdn_gating、l2norm 等，完整列表见 `third_party/ascend/unittest/kernels/vllm/` 目录。": "Currently covered vLLM kernels include: attention, rope, layer_norm, fused_gdn_gating, l2norm, etc. For the complete list, see the `third_party/ascend/unittest/kernels/vllm/` directory.",
        "**新增 Kernel 测试用例**：": "**Adding New Kernel Test Cases**:",
        "新增第三方 kernel 测试用例的流程：": "Process for adding new third-party kernel test cases:",
        "在 GPU 上准备 golden 数据（输入和期望输出），保存为 `.pt` 文件": "1. Prepare golden data (inputs and expected outputs) on GPU and save as `.pt` files",
        "在 `third_party/ascend/unittest/kernels/{library}/` 下新增 kernel 算子文件": "2. Add kernel operator files under `third_party/ascend/unittest/kernels/{library}/`",
        "将 `.pt` 文件上传至 OBS 桶：`https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/test/kernels/{library}_pt/{kernel_name}.pt`": "3. Upload `.pt` files to the OBS bucket: `https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/test/kernels/{library}_pt/{kernel_name}.pt`",
        "详细说明参见 `third_party/ascend/unittest/kernels/README.md`。": "For detailed instructions, see `third_party/ascend/unittest/kernels/README.md`.",
        "Fork-Pull 开发模式": "Fork-Pull Development Model",
        "1. Fork 仓库": "1. Fork the Repository",
        "在 GitHub 上 Fork [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) 到自己的账号下。": "Fork [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) to your own account on GitHub.",
        "2. 克隆并配置远程仓库": "2. Clone and Configure Remote Repository",
        "3. 创建开发分支": "3. Create Development Branch",
        "4. 开发与自测": "4. Development and Self-Testing",
        "开发完成后，请确保：": "After completing development, please ensure:",
        "代码通过 pre-commit 检查": "- Code passes pre-commit checks",
        "新增代码有对应的测试用例": "- New code has corresponding test cases",
        "所有相关测试通过": "- All related tests pass",
        "如修改了算子实现，建议运行 Inductor 验证和 kernel 对比测试": "- If operator implementations are modified, it's recommended to run Inductor validation and kernel comparison tests",
        "5. 提交代码": "5. Commit Code",
        "提交信息请遵循 [How to Write a Git Commit Message](https://cbea.ms/git-commit/#why-not-how) 规范。": "Please follow the [How to Write a Git Commit Message](https://cbea.ms/git-commit/#why-not-how) specification for commit messages.",
        "6. 创建 Pull Request": "6. Create Pull Request",
        "在 GitHub 上从你的 fork 仓库开发分支向 `triton-lang/triton-ascend` 的 `main` 分支创建 Pull Request。CI 流水线将自动运行。": "Create a Pull Request on GitHub from your fork repository's development branch to the `main` branch of `triton-lang/triton-ascend`. The CI pipeline will run automatically.",
        "PR 规范": "PR Guidelines",
        "PR 标题前缀": "PR Title Prefixes",
        "PR 标题建议使用以下前缀标明分类：": "PR titles are recommended to use the following prefixes to indicate categories:",
        "前缀": "Prefix",
        "说明": "Description",
        "`[BugFix]`": "`[BugFix]`",
        "Bug 修复": "Bug fix",
        "`[Kernel]`": "`[Kernel]`",
        "算子相关": "Operator related",
        "`[Core]`": "`[Core]`",
        "核心模块": "Core module",
        "`[Feature]`": "`[Feature]`",
        "新增特性": "New feature",
        "`[Refactor]`": "`[Refactor]`",
        "代码重构": "Code refactoring",
        "`[Revert]`": "`[Revert]`",
        "代码回滚": "Code revert",
        "`[Perf]`": "`[Perf]`",
        "性能优化": "Performance optimization",
        "`[Doc]`": "`[Doc]`",
        "文档更新": "Documentation update",
        "`[Test]`": "`[Test]`",
        "测试相关": "Test related",
        "`[CI]`": "`[CI]`",
        "CI/CD 相关": "CI/CD related",
        "`[Misc]`": "`[Misc]`",
        "其他杂项": "Other miscellaneous",
        "PR 检查清单": "PR Checklist",
        "[ ] 非琐碎修改（如非拼写修正）": "- [ ] Non-trivial changes (not just typo fixes)",
        "[ ] 遵循 [提交信息规范](https://cbea.ms/git-commit/#why-not-how)": "- [ ] Follow [commit message guidelines](https://cbea.ms/git-commit/#why-not-how)",
        "[ ] 已运行 `pre-commit run --from-ref origin/main --to-ref HEAD`": "- [ ] Ran `pre-commit run --from-ref origin/main --to-ref HEAD`",
        "[ ] 已添加测试用例，或说明无需测试的原因": "- [ ] Added test cases, or explained why tests are not needed",
        "[ ] 如添加 lit 测试，遵循 [MLIR 测试最佳实践](https://mlir.llvm.org/getting_started/TestingGuide/#filecheck-best-practices)": "- [ ] If adding lit tests, follow [MLIR Testing Best Practices](https://mlir.llvm.org/getting_started/TestingGuide/#filecheck-best-practices)",
        "PR 审核与合并": "PR Review and Merge",
        "PR 需获得至少 2 个 LGTM（Looks Good To Me）和 1 个 Approval 后方可合并": "- PRs require at least 2 LGTMs (Looks Good To Me) and 1 Approval before merging",
        "审核人不得在自己的 PR 上添加 LGTM": "- Reviewers cannot add LGTM to their own PRs",
        "创建 PR 后请尽快合并，以降低合并冲突风险": "- After creating a PR, please merge it promptly to reduce merge conflict risks",
        "Issue 规范": "Issue Guidelines",
        "提交 Issue": "Submitting Issues",
        "报告问题时，请包含以下信息：": "When reporting issues, please include the following information:",
        "软件版本（Triton-Ascend、Python、OS 等）": "- Software versions (Triton-Ascend, Python, OS, etc.)",
        "问题类型（Bug 报告还是功能请求）": "- Issue type (bug report or feature request)",
        "添加对应标签": "- Add corresponding labels",
        "问题描述：发生了什么？": "- Issue description: What happened?",
        "预期行为：应该发生什么？": "- Expected behavior: What should have happened?",
        "复现步骤：尽可能详细": "- Reproduction steps: Be as detailed as possible",
        "问题协作": "Issue Collaboration",
        "如发现未解决的 Issue 正是你打算处理的，请先在 Issue 下评论说明": "- If you find an open Issue that you plan to work on, please comment on the Issue first",
        "对于打开较久的 Issue，处理前请先确认问题是否仍然存在": "- For Issues that have been open for a while, please confirm the issue still exists before working on it",
        "如自行解决了自己报告的 Issue，关闭前请通知其他人": "- If you resolve an Issue you reported yourself, please notify others before closing",
        "注意事项": "Notes",
        "避免提交不相关的变更": "- Avoid committing unrelated changes",
        "保持提交历史简洁有序": "- Keep commit history clean and organized",
        "创建 PR 前请 rebase 上游最新代码": "- Please rebase with the latest upstream code before creating a PR",
        "对于 Bug 修复 PR，请链接所有相关 Issue": "- For bug fix PRs, please link all related Issues",
        "更多信息": "More Information",
        "[安全须知](../../SECURITYNOTE_zh.md)": "- [Security Note](../../SECURITYNOTE.md)",
        "[治理文档](governance.md)": "- [Governance](governance.md)",
        "[常见问题](../FAQ.md)": "- [FAQ](../FAQ.md)",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')

        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                # Check if current is a merged translation we need to fix
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"contributing.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process community/governance.po
# ============================================================
def fix_governance_po():
    filepath = os.path.join(BASE_DIR, 'community', 'governance.po')
    entries, raw = parse_po(filepath)

    translations = {
        "治理": "Governance",
        "使命": "Mission",
        "原则": "Principles",
        "治理机制": "Governance Mechanism",
        "贡献者（Contributor）": "Contributor",
        "**职责**：": "**Responsibilities**:",
        "协助新贡献者融入项目": "- Help new contributors integrate into the project",
        "处理并回应社区 Issue": "- Handle and respond to community Issues",
        "评审 RFC 和代码": "- Review RFCs and code",
        "**要求**：": "**Requirements**:",
        "至少完成一次贡献": "- Complete at least one contribution",
        "持续积极参与项目，参与形式包括提交 Issue、代码评审、提交代码以及参与社区活动": "- Continuously and actively participate in the project through submitting Issues, code reviews, code submissions, and participating in community activities",
        "**权限**：由 GitHub 仓库 `Triage` 角色授予，包括仓库读取和克隆、Issue 与 PR 提交权限。": "**Permissions**: Granted the GitHub repository `Triage` role, including read and clone access, and permission to submit Issues and PRs.",
        "维护者（Maintainer）": "Maintainer",
        "制定项目的愿景和使命，塑造技术方向，确保长期成功": "- Define the project's vision and mission, shape technical direction, and ensure long-term success",
        "拥有代码合并权限，主导路线图规划": "- Have code merge permissions, lead roadmap planning",
        "评审社区贡献，持续改进代码": "- Review community contributions and continuously improve code",
        "负责核心模块的代码质量与架构决策": "- Responsible for code quality and architectural decisions of core modules",
        "参与版本发布决策": "- Participate in release decisions",
        "指导贡献者完成代码贡献": "- Guide contributors in completing code contributions",
        "参与社区技术会议，对技术方向、大型方案和 Release 等事项进行讨论和投票": "- Participate in community technical meetings, discuss and vote on technical direction, large-scale proposals, and Release matters",
        "深刻理解 Triton 和 Triton-Ascend 的代码库": "- Deep understanding of the Triton and Triton-Ascend codebase",
        "承诺持续代码贡献，具备设计、开发和 PR 评审工作流能力": "- Commitment to continuous code contribution, with capabilities in design, development, and PR review workflows",
        "积极参与社区代码评审，交付至少一项主要功能并保持持续高质量贡献": "- Actively participate in community code reviews, deliver at least one major feature, and maintain continuous high-quality contributions",
        "积极处理 Issue、回应咨询、参与讨论": "- Actively handle Issues, respond to inquiries, and participate in discussions",
        "在项目中做出过重大技术贡献（如核心功能开发、架构设计、性能优化等）": "- Made significant technical contributions to the project (such as core feature development, architectural design, performance optimization, etc.)",
        "展现出良好的技术判断力和社区协作能力": "- Demonstrated good technical judgment and community collaboration skills",
        "提名与任免": "Nomination and Removal",
        "成员资格基于个人能力，通过贡献、评审和讨论来体现": "- Membership is based on individual capability, demonstrated through contributions, reviews, and discussions",
        "候选人须认同 Triton-Ascend 的原则并保持行为一致": "- Candidates must agree with Triton-Ascend's principles and maintain consistent behavior",
        "连续 **6 个月** 未参与项目贡献的 Maintainer 可转为荣誉状态（Emeritus）": "- Maintainers who have not contributed to the project for **6 consecutive months** may be transitioned to Emeritus status",
        "成员资格授予个人，不与雇主绑定": "- Membership is granted to individuals, not tied to employers",
        "提名流程": "Nomination Process",
        "提名可由任何人发起，包括自我提名，以社区issue的方式进行提名": "- Nominations can be initiated by anyone, including self-nominations, through a community Issue",
        "提名时需提供候选人的贡献记录和技术能力说明": "- Nominations must include the candidate's contribution record and technical capability description",
        "现有 Maintainer 审核评估，需**获得当前Maintainer半数及以上同意并无人反对**": "- Existing Maintainers review and evaluate, requiring **more than half of current Maintainers to agree with no objections**",
        "定期审视": "Regular Review",
        "**每月**在技术例会上对所有 Maintainer 的活跃度进行审视": "- **Monthly** review of all Maintainers' activity during technical meetings",
        "连续**6 个月**未参与项目贡献的 Maintainer，经讨论后可转为荣誉状态（Emeritus）": "- Maintainers who have not contributed to the project for **6 consecutive months** may be transitioned to Emeritus status after discussion",
        "Emeritus Maintainer 可随时通过重新活跃贡献申请恢复 Maintainer 身份": "- Emeritus Maintainers can apply to restore their Maintainer status at any time by resuming active contributions",
        "维护者与贡献者": "Maintainers and Contributors",
        "姓名": "Name",
        "ID": "ID",
        "加入时间": "Join Date",
        "Chen Cheng": "Chen Cheng",
        "ccdedreams": "ccdedreams",
        "2025-09-27": "2025-09-27",
        "Lijuan Hai": "Lijuan Hai",
        "HaiLijuan": "HaiLijuan",
        "Tao Wang": "Tao Wang",
        "wangtao489": "wangtao489",
        "Yingcong Liang": "Yingcong Liang",
        "wcleungaj": "wcleungaj",
        "Jingchang Shi": "Jingchang Shi",
        "shijingchang": "shijingchang",
        "Hao Zhou": "Hao Zhou",
        "zhouhao176": "zhouhao176",
        "Kaipeng Xing": "Kaipeng Xing",
        "kpxing": "kpxing",
        "Zhijie Zhao": "Zhijie Zhao",
        "zhaozhijie": "zhaozhijie",
        "Kaixin Yang": "Kaixin Yang",
        "yangkaixin": "yangkaixin",
        "Chunli Zhang": "Chunli Zhang",
        "zhang-chunli01": "zhang-chunli01",
        "Ce Zhu": "Ce Zhu",
        "zhucehw": "zhucehw",
        "2025-12-23": "2025-12-23",
        "Xuan Peng": "Xuan Peng",
        "HinPeng": "HinPeng",
        "Ziqi Hong": "Ziqi Hong",
        "candyhong": "candyhong",
        "2026-03-26": "2026-03-26",
        "需要": "No.",
        "贡献者": "Contributor",
        "首次提交日期": "First Contribution Date",
        "PR": "PR",
        "149": "149",
        "[ggppff24](https://gitcode.com/ggppff24)": "[ggppff24](https://gitcode.com/ggppff24)",
        "2026-05-06": "2026-05-06",
        "[#1769](https://gitcode.com/Ascend/triton-ascend/merge_requests/1769)": "[#1769](https://gitcode.com/Ascend/triton-ascend/merge_requests/1769)",
        "148": "148",
        "[wxlong_ustc](https://gitcode.com/wxlong_ustc)": "[wxlong_ustc](https://gitcode.com/wxlong_ustc)",
        "2026-04-29": "2026-04-29",
        "[#1733](https://gitcode.com/Ascend/triton-ascend/merge_requests/1733)": "[#1733](https://gitcode.com/Ascend/triton-ascend/merge_requests/1733)",
        "147": "147",
        "[liwei571](https://gitcode.com/liwei571)": "[liwei571](https://gitcode.com/liwei571)",
        "[#1740](https://gitcode.com/Ascend/triton-ascend/merge_requests/1740)": "[#1740](https://gitcode.com/Ascend/triton-ascend/merge_requests/1740)",
        "146": "146",
        "[WangYiMou](https://gitcode.com/WangYiMou)": "[WangYiMou](https://gitcode.com/WangYiMou)",
        "2026-04-28": "2026-04-28",
        "[#1697](https://gitcode.com/Ascend/triton-ascend/merge_requests/1697)": "[#1697](https://gitcode.com/Ascend/triton-ascend/merge_requests/1697)",
        "145": "145",
        "[Four1er](https://gitcode.com/Four1er)": "[Four1er](https://gitcode.com/Four1er)",
        "[#1742](https://gitcode.com/Ascend/triton-ascend/merge_requests/1742)": "[#1742](https://gitcode.com/Ascend/triton-ascend/merge_requests/1742)",
        "144": "144",
        "[gcw_LcYTzRJy](https://gitcode.com/gcw_LcYTzRJy)": "[gcw_LcYTzRJy](https://gitcode.com/gcw_LcYTzRJy)",
        "2026-04-27": "2026-04-27",
        "[#1690](https://gitcode.com/Ascend/triton-ascend/merge_requests/1690)": "[#1690](https://gitcode.com/Ascend/triton-ascend/merge_requests/1690)",
        "143": "143",
        "[zhaojiqiao](https://gitcode.com/zhaojiqiao)": "[zhaojiqiao](https://gitcode.com/zhaojiqiao)",
        "[#1722](https://gitcode.com/Ascend/triton-ascend/merge_requests/1722)": "[#1722](https://gitcode.com/Ascend/triton-ascend/merge_requests/1722)",
        "142": "142",
        "[andreybokhanko](https://gitcode.com/andreybokhanko)": "[andreybokhanko](https://gitcode.com/andreybokhanko)",
        "[#1675](https://gitcode.com/Ascend/triton-ascend/merge_requests/1675)": "[#1675](https://gitcode.com/Ascend/triton-ascend/merge_requests/1675)",
        "141": "141",
        "[weizhan4](https://gitcode.com/weizhan4)": "[weizhan4](https://gitcode.com/weizhan4)",
        "2026-04-25": "2026-04-25",
        "[#1702](https://gitcode.com/Ascend/triton-ascend/merge_requests/1702)": "[#1702](https://gitcode.com/Ascend/triton-ascend/merge_requests/1702)",
        "140": "140",
        "[Sky_miner](https://gitcode.com/Sky_miner)": "[Sky_miner](https://gitcode.com/Sky_miner)",
        "2026-04-24": "2026-04-24",
        "[#1680](https://gitcode.com/Ascend/triton-ascend/merge_requests/1680)": "[#1680](https://gitcode.com/Ascend/triton-ascend/merge_requests/1680)",
        "139": "139",
        "[zhangchaofan](https://gitcode.com/zhangchaofan)": "[zhangchaofan](https://gitcode.com/zhangchaofan)",
        "2026-04-22": "2026-04-22",
        "[#1265](https://gitcode.com/Ascend/triton-ascend/merge_requests/1265)": "[#1265](https://gitcode.com/Ascend/triton-ascend/merge_requests/1265)",
        "138": "138",
        "[yuanjingya](https://gitcode.com/yuanjingya)": "[yuanjingya](https://gitcode.com/yuanjingya)",
        "[#1638](https://gitcode.com/Ascend/triton-ascend/merge_requests/1638)": "[#1638](https://gitcode.com/Ascend/triton-ascend/merge_requests/1638)",
        "137": "137",
        "[shi-yufeng99](https://gitcode.com/shi-yufeng99)": "[shi-yufeng99](https://gitcode.com/shi-yufeng99)",
        "2026-04-13": "2026-04-13",
        "[#1570](https://gitcode.com/Ascend/triton-ascend/merge_requests/1570)": "[#1570](https://gitcode.com/Ascend/triton-ascend/merge_requests/1570)",
        "136": "136",
        "[zaq15csdn](https://gitcode.com/zaq15csdn)": "[zaq15csdn](https://gitcode.com/zaq15csdn)",
        "2026-04-11": "2026-04-11",
        "[#1593](https://gitcode.com/Ascend/triton-ascend/merge_requests/1593)": "[#1593](https://gitcode.com/Ascend/triton-ascend/merge_requests/1593)",
        "135": "135",
        "[wuyao51511](https://gitcode.com/wuyao51511)": "[wuyao51511](https://gitcode.com/wuyao51511)",
        "2026-04-10": "2026-04-10",
        "[#1592](https://gitcode.com/Ascend/triton-ascend/merge_requests/1592)": "[#1592](https://gitcode.com/Ascend/triton-ascend/merge_requests/1592)",
        "134": "134",
        "[jtaizhang](https://gitcode.com/jtaizhang)": "[jtaizhang](https://gitcode.com/jtaizhang)",
        "2026-04-08": "2026-04-08",
        "[#1575](https://gitcode.com/Ascend/triton-ascend/merge_requests/1575)": "[#1575](https://gitcode.com/Ascend/triton-ascend/merge_requests/1575)",
        "133": "133",
        "[jjin_750629](https://gitcode.com/jjin_750629)": "[jjin_750629](https://gitcode.com/jjin_750629)",
        "2026-04-07": "2026-04-07",
        "[#1498](https://gitcode.com/Ascend/triton-ascend/merge_requests/1498)": "[#1498](https://gitcode.com/Ascend/triton-ascend/merge_requests/1498)",
        "132": "132",
        "[lbqg](https://gitcode.com/lbqg)": "[lbqg](https://gitcode.com/lbqg)",
        "2026-04-02": "2026-04-02",
        "[#1506](https://gitcode.com/Ascend/triton-ascend/merge_requests/1506)": "[#1506](https://gitcode.com/Ascend/triton-ascend/merge_requests/1506)",
        "131": "131",
        "[huangyujun123](https://gitcode.com/huangyujun123)": "[huangyujun123](https://gitcode.com/huangyujun123)",
        "2026-03-31": "2026-03-31",
        "[#1505](https://gitcode.com/Ascend/triton-ascend/merge_requests/1505)": "[#1505](https://gitcode.com/Ascend/triton-ascend/merge_requests/1505)",
        "130": "130",
        "[danielcm585](https://gitcode.com/danielcm585)": "[danielcm585](https://gitcode.com/danielcm585)",
        "[#1444](https://gitcode.com/Ascend/triton-ascend/merge_requests/1444)": "[#1444](https://gitcode.com/Ascend/triton-ascend/merge_requests/1444)",
        "129": "129",
        "[OliverCWY](https://gitcode.com/OliverCWY)": "[OliverCWY](https://gitcode.com/OliverCWY)",
        "[#1441](https://gitcode.com/Ascend/triton-ascend/merge_requests/1441)": "[#1441](https://gitcode.com/Ascend/triton-ascend/merge_requests/1441)",
        "128": "128",
        "[cxtverygood123](https://gitcode.com/cxtverygood123)": "[cxtverygood123](https://gitcode.com/cxtverygood123)",
        "[#1459](https://gitcode.com/Ascend/triton-ascend/merge_requests/1459)": "[#1459](https://gitcode.com/Ascend/triton-ascend/merge_requests/1459)",
        "127": "127",
        "[shaoyiyang](https://gitcode.com/shaoyiyang)": "[shaoyiyang](https://gitcode.com/shaoyiyang)",
        "2026-03-21": "2026-03-21",
        "[#1429](https://gitcode.com/Ascend/triton-ascend/merge_requests/1429)": "[#1429](https://gitcode.com/Ascend/triton-ascend/merge_requests/1429)",
        "126": "126",
        "[xiechenye](https://gitcode.com/xiechenye)": "[xiechenye](https://gitcode.com/xiechenye)",
        "2026-03-20": "2026-03-20",
        "[#1407](https://gitcode.com/Ascend/triton-ascend/merge_requests/1407)": "[#1407](https://gitcode.com/Ascend/triton-ascend/merge_requests/1407)",
        "125": "125",
        "[wuzw_05](https://gitcode.com/wuzw_05)": "[wuzw_05](https://gitcode.com/wuzw_05)",
        "[#1418](https://gitcode.com/Ascend/triton-ascend/merge_requests/1418)": "[#1418](https://gitcode.com/Ascend/triton-ascend/merge_requests/1418)",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')

        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"governance.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process community/release_policy.po
# ============================================================
def fix_release_policy_po():
    filepath = os.path.join(BASE_DIR, 'community', 'release_policy.po')
    entries, raw = parse_po(filepath)

    translations = {
        "版本策略": "Release Policy",
        "版本号规则": "Version Numbering",
        "**MAJOR.MINOR**：与上游 Triton 版本一一对应，如 Triton-Ascend `3.2` 基于 Triton `3.2`": "- **MAJOR.MINOR**: Corresponds one-to-one with upstream Triton versions, e.g., Triton-Ascend `3.2` is based on Triton `3.2`",
        "**PATCH**：Triton-Ascend 的 `PATCH` 版本可能多于上游 Triton，用于做 `MAJOR.MINOR` 级别的问题修复或改进，例如 Triton-Ascend `3.2.0` 和 `3.2.1` 均是基于 Triton `3.2.0`": "- **PATCH**: Triton-Ascend's `PATCH` version may be higher than upstream Triton, used for issue fixes or improvements at the `MAJOR.MINOR` level, e.g., both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`",
        "**rcN**：预发布候选版本（Release Candidate），按需发布，供社区提前测试和反馈": "- **rcN**: Release Candidate, published as needed for early testing and feedback from the community",
        "**postN**：已发布版本的问题修复版本（Post Release），按需发布，针对已有稳定版本的补丁": "- **postN**: Post-release patches for already released versions, published as needed to address issues in stable versions",
        "版本分支策略": "Branching Strategy",
        "`main` 分支为最新开发分支，跟踪上游 Triton 最新版本": "- The `main` branch is the latest development branch, tracking the latest upstream Triton version",
        "每个发布版本创建对应的版本开发分支（如 `releases/v3.2.x`），该分支与社区拉取 release 的 commit id 相同": "- Each release version creates a corresponding release development branch (e.g., `releases/v3.2.x`), which has the same commit id as the community release",
        "功能开发应在 fork 仓库中完成，通过 `PR` 合并至 Triton-Ascend 仓库": "- Feature development should be done in fork repositories and merged into the Triton-Ascend repository via `PR`",
        "**`main` 分支对应关系：**": "**`main` Branch Mapping:**",
        "Triton-Ascend": "Triton-Ascend",
        "Triton commit hash": "Triton commit hash",
        "Python": "Python",
        "CANN": "CANN",
        "PyTorch": "PyTorch",
        "LLVM commit hash": "LLVM commit hash",
        "Patch": "Patch",
        "`main`": "`main`",
        "`3.9~3.13`": "`3.9~3.13`",
        "`9.0.0`": "`9.0.0`",
        "`2.7.1`": "`2.7.1`",
        "维护分支与生命周期": "Maintenance Branches and Lifecycle",
        "维护分支的状态分为：": "Maintenance branch statuses include:",
        "**活跃（`Active`）**：持续接受 Bug 修复、功能改进和安全补丁，会继续特性演进或发布新版本": "- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; will continue to evolve features or release new versions",
        "**维护（`Maintenance`）**：仅接受关键 Bug 修复和安全补丁，不再发布功能改进": "- **Maintenance**: Only accepts critical bug fixes and security patches; no longer releases feature improvements",
        "**归档（`End of Life`）**：不再接受任何修复，分支停止维护": "- **End of Life**: No longer accepts any fixes; branch maintenance has stopped",
        "分支": "Branch",
        "状态": "Status",
        "对应 Triton 版本": "Triton Version",
        "Triton-Ascend 发布版本": "Triton-Ascend Release",
        "维护截止日期": "End of Maintenance",
        "`活跃`": "`Active`",
        "`3.5.0`": "`3.5.0`",
        "/": "/",
        "`releases/v3.2.1`": "`releases/v3.2.1`",
        "`3.2.0`": "`3.2.0`",
        "`3.2.1`": "`3.2.1`",
        "`releases/v3.1.x`": "`releases/v3.1.x`",
        "`维护`": "`Maintenance`",
        "`3.2.0rc2`，`3.2.0rc3`，`3.2.0rc4`，`3.2.0`": "`3.2.0rc2`, `3.2.0rc3`, `3.2.0rc4`, `3.2.0`",
        "发布周期": "Release Cycle",
        "**稳定版本**：根据项目版本节奏发布，并非每个上游 Triton 版本都发布对应的稳定版本": "- **Stable releases**: Released according to project version cadence, not every upstream Triton version will have a corresponding stable release",
        "**rc 版本**：与上游 Triton 版本节奏统一发布，供早期用户测试": "- **rc releases**: Released in sync with upstream Triton version cadence for early user testing",
        "**post 版本**：按需发布，针对已有稳定版本的问题修复": "- **post releases**: Released as needed to address issues in existing stable versions",
        "发布时间线": "Release Timeline",
        "日期": "Date",
        "事件": "Event",
        "2025-05-26": "2025-05-26",
        "发布预览版本 `3.2.0rc2`": "Released preview version `3.2.0rc2`",
        "2025-11-12": "2025-11-12",
        "发布预览版本 `3.2.0rc3`": "Released preview version `3.2.0rc3`",
        "2025-11-14": "2025-11-14",
        "发布预览版本 `3.2.0rc4`": "Released preview version `3.2.0rc4`",
        "2026-01-21": "2026-01-21",
        "发布正式版本 `3.2.0`": "Released stable version `3.2.0`",
        "2026-05-06": "2026-05-06",
        "发布正式版本 `3.2.1`": "Released stable version `3.2.1`",
        "版本兼容性矩阵": "Version Compatibility Matrix",
        "Triton": "Triton",
        "LLVM Patch": "LLVM Patch",
        "`3.9`(x86), `3.10-3.13`": "`3.9`(x86), `3.10-3.13`",
        "`b5cc222`": "`b5cc222`",
        "-": "-",
        "`3.9-3.11`": "`3.9-3.11`",
        "`8.5.0`": "`8.5.0`",
        "`2.6.0`": "`2.6.0`",
        "`3.2.0rc4`": "`3.2.0rc4`",
        "`3.2.0rc3`": "`3.2.0rc3`",
        "`86b69c3`": "`86b69c3`",
        "`3.2.0rc2`": "`3.2.0rc2`",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')

        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"release_policy.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process community/community_technical_meeting.po
# ============================================================
def fix_community_technical_meeting_po():
    filepath = os.path.join(BASE_DIR, 'community', 'community_technical_meeting.po')
    entries, raw = parse_po(filepath)

    translations = {
        "社区技术会议": "Community Technical Meeting",
        "会议内容": "Meeting Content",
        "技术方向讨论与决策": "- Technical direction discussion and decision-making",
        "大型方案（RFC）评审与投票": "- Large-scale proposal (RFC) review and voting",
        "版本发布计划与时间线": "- Version release planning and timeline",
        "社区 Issue 和 PR 的优先级排序": "- Community Issue and PR priority ranking",
        "贡献者提名与任免讨论": "- Contributor nomination and removal discussions",
        "会议安排": "Meeting Schedule",
        "**例会频率**：双周": "- **Meeting Frequency**: Bi-weekly",
        "**会议时间**：周五15:00": "- **Meeting Time**: Friday 15:00",
        "**会议链接**：待公布": "- **Meeting Link**: To be announced",
        "**会议纪要**：待公布": "- **Meeting Minutes**: To be announced",
        "**议题提交**：待公布": "- **Agenda Submission**: To be announced",
        "投票规则": "Voting Rules",
        "未参会的 `Maintainer` 需在会议后 3 个工作日内反馈投票意见，逾期未投票视为弃权": "- Maintainers who do not attend the meeting must provide their vote within 3 business days after the meeting; failure to vote within the deadline is considered abstention",
        "投票以参与投票的 `Maintainer` 过半数同意即为通过（弃权不计入票数）": "- A vote is passed when more than half of the participating Maintainers agree (abstentions are not counted in the vote total)",
        "投票结果记录在会议纪要中": "- Voting results are recorded in the meeting minutes",
        "决策流程": "Decision-Making Process",
        "常规决策": "Routine Decisions",
        "重大决策": "Major Decisions",
        "以下事项视为重大决策，需在社区技术会议上由 `Maintainer` 集体讨论决定：": "The following matters are considered major decisions and require collective discussion by Maintainers at community technical meetings:",
        "架构层面的重大变更": "- Major architectural changes",
        "`API` 的破坏性变更": "- Breaking API changes",
        "新增核心功能或移除已有功能": "- Adding core features or removing existing features",
        "第三方依赖的重大升级": "- Major upgrades to third-party dependencies",
        "版本发布计划": "- Version release planning",
        "重大决策的流程：": "Major decision process:",
        "提交 `RFC`（Request for Comments），描述变更内容、影响范围和实施方案": "1. Submit an RFC (Request for Comments), describing the change content, impact scope, and implementation plan",
        "在技术会议上进行讨论": "2. Discuss at the technical meeting",
        "`Maintainer` 共同决策": "3. Maintainers make a joint decision",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')

        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"community_technical_meeting.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process programming_guide/index.po
# ============================================================
def fix_programming_guide_index_po():
    filepath = os.path.join(BASE_DIR, 'programming_guide', 'index.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Triton算子开发指南": "Triton Operator Development Guide",
        "文档组织": "Documentation Structure",
        "本指南将通用开发原则和按硬件执行单元划分的算子开发路径分开组织：": "This guide separates common development rules from operator-specific development paths:",
        "本页介绍所有 Triton-Ascend 算子都需要关注的通用问题，包括分核、片上内存、访存、Tiling 和 Autotune。": "- This page covers common Triton-Ascend concerns, including core allocation, on-chip memory, memory access, tiling, and autotune.",
        "[Vector 算子开发](./vector_operator.md) 介绍主要由 Vector Core 执行的逐元素、归约、Gather/Scatter 等算子。": "- [Vector Operator Development](./vector_operator.md) describes element-wise, reduction, gather/scatter, and other operators mainly executed on Vector Cores.",
        "[Cube 算子开发](./cube_operator.md) 介绍以 `tl.dot`、矩阵乘、批量矩阵乘为核心的算子。": "- [Cube Operator Development](./cube_operator.md) describes operators whose main computation is `tl.dot`, matrix multiplication, or batched matrix multiplication.",
        "[CV 融合算子开发](./cv_fusion_operator.md) 介绍同一个算子中同时存在 Cube 计算和 Vector 后处理、归约、Softmax 或跨核协同的场景。": "- [CV Fusion Operator Development](./cv_fusion_operator.md) describes operators that combine Cube computation with Vector post-processing, reductions, softmax, or cross-core coordination.",
        "通用多核任务并行": "Common Multi-Core Task Parallelism",
        "设置最大硬件核数": "Setting the Maximum Number of Hardware Cores",
        "对于纯Vector算子，分核数等于**Vector核数量**": "* For pure vector operators, the number of cores is equal to the **number of vector cores**.",
        "对于CV融合算子，分核数等于**Cube核数量**（通常为Vector核数量的一半），算子执行时会按1：2的比例调用Vector核": "* For CV fusion operators, the number of cores is equal to the **number of cube cores** (usually half of the number of vector cores). During operator execution, vector cores are called at a ratio of 1:2.",
        "通用单核数据搬运": "Common Single-Core Data Transfer",
        "设置合适的循环内数据分块大小（BLOCK SIZE）": "Setting the Proper Data Block Size (BLOCK SIZE)",
        "尽量保证Tensor的尾轴大小数据对齐": "Aligning the Size of the Tail Axis of the Tensor",
        "示例": "Example",
        "优化前后性能分析和对比": "Performance analysis and comparison before and after optimization",
        "存算并行": "Parallel Storage and Computation",
        "Triton-Ascend支持两种数据处理模式：存算串行和存算并行。": "Triton-Ascend supports two data processing modes: serial storage and computation and parallel storage and computation.",
        "Tiling优化": "Tiling Optimization",
        "Triton Autotune 自动调优": "Triton Autotune",
        "如何在NPU上避免UB OVERFLOW": "How Do I Avoid UB Overflow on the NPU?",
        "通用单核数据运算": "Common Single-Core Data Computation",
        "开发目标": "R&D Goals",
        "开发步骤": "Development Procedure",
        "Op Name": "Op Name",
        "aiv_mte2_time(us)": "aiv_mte2_time(us)",
        "aiv_mte2_ratio": "aiv_mte2_ratio",
        "未优化": "Unoptimized",
        "pick_kernel": "pick_kernel",
        "0.686": "0.686",
        "0.008": "0.008",
        "优化": "Optimized",
        "1.041": "1.041",
        "0.066": "0.066",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')

        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"programming_guide/index.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process programming_guide/vector_operator.po
# ============================================================
def fix_vector_operator_po():
    filepath = os.path.join(BASE_DIR, 'programming_guide', 'vector_operator.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Vector 算子开发": "Vector Operator Development",
        "Vector 算子主要由 Vector Core 执行，典型形态包括逐元素计算、行级归约、类型转换、Gather/Scatter、Mask 更新以及不含 `tl.dot` 的小型融合算子。开发重点不是把 grid 切得越细越好，而是在固定物理 Vector Core 数量的前提下，让每个 program 在核内循环处理多个 tile。": "Vector operators are mainly executed by Vector Cores. Typical examples include element-wise computation, row-wise reduction, type conversion, gather/scatter, masked update, and small fused operators without `tl.dot`. The key is not to create as many grid programs as possible, but to keep the launch close to the number of physical Vector Cores and let each program process multiple tiles in an inner loop.",
        "Vector 简单算子开发": "Simple Vector Operator Development",
        "简单 Vector 算子可以从本仓的 [向量相加样例](../examples/01_vector_add_example.md) 或 `third_party/ascend/tutorials/01-vector-add.py` 入手。该类算子的基本步骤如下：": "For a simple Vector operator, start with the [Vector Addition example](../examples/01_vector_add_example.md) or `third_party/ascend/tutorials/01-vector-add.py`. The basic pattern is:",
        "用 `tl.arange` 构造当前 tile 的连续偏移。": "1. Build contiguous offsets for the current tile with `tl.arange`.",
        "用 `mask` 保护尾块，避免越界 load/store。": "2. Use a tail mask to guard load/store.",
        "完成逐元素计算后写回结果。": "3. Compute the element-wise expression and store the result.",
        "当 grid 数远大于物理核数时，将 grid 固定为 `num_vectorcore`，在 kernel 内用 `range(pid, num_blocks, num_core)` 分批处理。": "4. If the grid is much larger than the physical core count, set the grid to `num_vectorcore` and process tiles with `range(pid, num_blocks, num_core)` inside the kernel.",
        "基础 kernel 结构如下：": "Core structure is as follows:",
        "开发时优先检查三类问题：": "When developing, check these three aspects first:",
        "**数据类型**：Ascend Vector 单元对不同整数类型的支持和性能不同。对于不影响精度的索引、长度、偏移类数据，优先使用 `int32`，可参考 `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` 和 `002-vector_cmp.zh.md`。": "- **Data type**: Ascend Vector units have different performance for integer types. Prefer `int32` for indices, lengths, and offsets when precision allows. See `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` and `002-vector_cmp.zh.md`.",
        "**BLOCK_SIZE**：BLOCK_SIZE 需要在 UB 容量内尽量大。若出现 UB overflow，先降低单次处理元素数，再考虑拆分子块。": "- **BLOCK_SIZE**: Keep it as large as possible without exceeding UB capacity. If UB overflow occurs, reduce the tile size or split it into sub-blocks.",
        "**分核数**：NPU 物理 Vector Core 数量通常为几十个。小 tile 大 grid 的 GPU 写法迁移到 NPU 时，容易因多轮下发带来明显开销。": "- **Core count**: GPU-style small tiles with very large grids often cause repeated dispatch overhead on NPUs.",
        "Vector 复杂算子开发": "Complex Vector Operator Development",
        "复杂 Vector 算子通常不是单个逐元素表达式，而是带有离散访存、批量重排、多个输出或长 hidden size 的组合逻辑。可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的以下案例：": "Complex Vector operators usually combine irregular memory access, token reordering, multiple outputs, or long hidden dimensions. Useful references in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) include:",
        "[`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py)：Megablocks gather/scatter/scatter_wgrad 的 Ascend 亲和实现。": "- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py): An Ascend-friendly implementation of Megablocks gather/scatter/scatter_wgrad.",
        "[`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py)：按 expert/bin 分组后的 gather/scatter。": "- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py): Gather/scatter grouped by expert/bin.",
        "[`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py)：带 padding 的 MoE gather/scatter。": "- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py): MoE gather/scatter with padding.",
        "这类算子的组织方式通常是：": "Complex Vector operators are typically organized as follows:",
        "**按物理核切分外层任务**：用 `num_vectorcore` 作为 grid，每个 program 负责一段 indices 或 token。": "1. **Split outer tasks by physical Vector Core count**: Use `num_vectorcore` as the grid, with each program responsible for a range of indices or tokens.",
        "**按 UB 容量切分 hidden 维**：对 `NUM_COLUMNS` 使用 `BLOCK_X` 分块，并预留 double buffer、索引和临时张量的空间。": "2. **Split the hidden dimension by UB capacity**: Use `BLOCK_X` for `NUM_COLUMNS` and reserve space for double buffers, indices, and temporary tensors.",
        "**用 `SUB_BLOCK_SIZE` 合并小粒度离散任务**：一次加载一组 indices，在 UB 中组织成连续临时块，减少 GM 标量访存和多次 store。": "3. **Use `SUB_BLOCK_SIZE` to batch small irregular tasks**: Load a group of indices at once, organize them into contiguous temporary blocks in UB, reducing GM scalar access and multiple stores.",
        "**用扩展语义管理 UB 内局部数据**：使用 `tl.insert_slice` 合并多行数据，使用 `tl.extract_slice` 取出子块后再分散写回。": "4. **Use extended semantics for UB-local data management**: Use `tl.insert_slice` to assemble multiple rows and `tl.extract_slice` to extract sub-blocks before scattering writes.",
        "**为尾块保留统一 mask**：复杂 gather/scatter 中同时存在 index mask、column mask 和 expert/bin 边界，建议分别命名并只在 load/store 处组合。": "5. **Keep unified masks for tail blocks**: In complex gather/scatter, index masks, column masks, and expert/bin boundary masks may coexist. Name them separately and combine only at load/store sites.",
        "典型的 UB 预算思路如下：": "A typical UB budget estimation is as follows:",
        "当复杂 Vector 算子性能不达预期时，优先从以下方向排查：": "When complex Vector operator performance is poor, check the following aspects first:",
        "grid 是否远大于物理 Vector Core 数，导致多轮下发。": "- Whether the grid is much larger than the physical Vector Core count, causing multiple dispatch rounds.",
        "离散访存是否可转化为“批量搬入 UB 后在 UB 内选择”。": "- Whether irregular access can be converted into \"bulk load to UB and select in UB\".",
        "尾轴是否满足 32B 对齐；不满足时是否可用转置或借轴转置规避自动 padding。": "- Whether the tail axis is 32B aligned; if not, whether transpose or axis borrowing can avoid automatic padding.",
        "`BLOCK_X` 和 `SUB_BLOCK_SIZE` 是否造成 UB overflow 或过小的搬运粒度。": "- Whether `BLOCK_X` or `SUB_BLOCK_SIZE` causes UB overflow or too-small transfer granularity.",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"vector_operator.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process programming_guide/cube_operator.po
# ============================================================
def fix_cube_operator_po():
    filepath = os.path.join(BASE_DIR, 'programming_guide', 'cube_operator.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Cube 算子开发": "Cube Operator Development",
        "Cube 算子以矩阵乘或批量矩阵乘为主要计算负载，Triton 代码中通常以 `tl.dot` 为核心。Cube 算子的关键是围绕 M/N/K 三个维度设计 tile，使 A/B tile 能高效搬运到片上并在 Cube Core 上完成累加。": "Cube operators use matrix multiplication or batched matrix multiplication as the main workload. In Triton code, the core operation is usually `tl.dot`. The main task is to design M/N/K tiles so that A and B tiles can be moved on chip efficiently and accumulated on Cube Cores.",
        "Cube 简单算子开发": "Simple Cube Operator Development",
        "简单 Cube 算子可参考本仓 [矩阵乘法样例](../examples/05_matrix_multiplication_example.md) 或 `third_party/ascend/tutorials/03-matrix-multiplication.py`。一个最小开发路径包括：": "For a simple Cube operator, refer to the [Matrix Multiplication example](../examples/05_matrix_multiplication_example.md) or `third_party/ascend/tutorials/03-matrix-multiplication.py`.",
        "明确输入输出 shape 和 stride，例如 `A[M, K]`、`B[K, N]`、`C[M, N]`。": "1. Define input/output shapes and strides, for example `A[M, K]`, `B[K, N]`, and `C[M, N]`.",
        "用 `tl.program_id` 映射当前 program 到输出矩阵的 `(pid_m, pid_n)` tile。": "2. Map `tl.program_id` to the output tile `(pid_m, pid_n)`.",
        "用 `BLOCK_SIZE_M/N/K` 构造 A/B 的二维偏移。": "3. Build 2D offsets for A and B using `BLOCK_SIZE_M/N/K`.",
        "沿 K 维循环加载 A/B 子块，并用 `tl.dot` 累加到 fp32 accumulator。": "4. Loop over K, load A/B sub-blocks, and accumulate with `tl.dot` in fp32.",
        "将 accumulator 转为输出 dtype，并用边界 mask 写回 C。": "5. Cast the accumulator to the output dtype and store with boundary masks.",
        "核心结构如下：": "Core structure is as follows:",
        "简单 Cube 算子调参时优先关注：": "When tuning simple Cube operators, check the following first:",
        "`BLOCK_M/N/K` 是否满足硬件支持和 UB/L1 容量限制。": "- Whether `BLOCK_M/N/K` meets hardware support and UB/L1 capacity limits.",
        "K 维循环是否可以开启 `multibuffer` 以形成搬运和计算流水。": "- Whether the K loop can enable `multibuffer` to pipeline data movement and computation.",
        "输出 tile 是否包含额外 bias、scale、activation。如果后处理很轻，可以仍归为 Cube 算子；如果后处理包含明显 Vector 归约或跨核同步，应按 CV 融合算子组织。": "- Whether the output tile includes extra bias, scale, or activation. If the post-processing is lightweight, it can still be classified as a Cube operator; if it involves significant Vector reduction or cross-core synchronization, organize it as a CV fusion operator.",
        "Cube 复杂算子开发": "Complex Cube Operator Development",
        "复杂 Cube 场景通常来自 attention、batched matmul、grouped matmul 或形状不规则的矩阵乘。当前 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 主分支的复杂案例集中在 `tutorial/best_practice/`，其中 [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) 可以作为复杂 Cube 核心逻辑的参考：它包含 QK、PV 两段 `tl.dot`，并展示了 KV cache 离散索引下如何重组 K/V 访存。": "Complex Cube cases often come from attention, batched matmul, grouped matmul, or irregular shapes. In the current main branch of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops), complex cases are mainly in `tutorial/best_practice/`. [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) is a useful reference for the Cube core because it contains QK and PV `tl.dot` stages and shows how to reorganize K/V memory access under discrete KV-cache indices.",
        "复杂 Cube 算子建议按以下顺序拆解：": "Complex Cube operators should be decomposed in the following order:",
        "**先抽出纯矩阵乘核心**：确认每次 `tl.dot` 的输入 tile shape、dtype、累加 dtype 和输出 tile shape。": "1. **Extract the pure matmul core first**: Confirm the input tile shape, dtype, accumulator dtype, and output tile shape for each `tl.dot`.",
        "**再处理不规则访存**：如果 K/V cache 低维离散、高维连续，直接二维 load 可能退化为标量访存。可先按连续维搬入 UB，再通过转置或 `tl.insert_slice` 重组为 `tl.dot` 需要的布局。": "2. **Handle irregular memory access next**: If K/V cache is discrete in the low dimension and continuous in the high dimension, direct 2D load may degrade to scalar access. Load by continuous dimension into UB first, then reorganize into the layout required by `tl.dot` via transpose or `tl.insert_slice`.",
        "**把归约和归一化留到边界明确的位置**：例如 attention 中的 `max/sum/exp` 属于 Vector 逻辑，若和 `tl.dot` 放在同一 kernel，需要转到 [CV 融合算子开发](./cv_fusion_operator.md) 的思路。": "3. **Leave reductions and normalization to well-defined boundaries**: For example, `max/sum/exp` in attention belongs to Vector logic. If placed in the same kernel as `tl.dot`, refer to [CV Fusion Operator Development](./cv_fusion_operator.md).",
        "**为长 K 或长序列设计内层循环**：K 维循环要控制单次 A/B tile 的片上占用；序列维循环要避免一次 load 过大的 K/V block。": "4. **Design inner loops for long K or long sequences**: The K loop should control the on-chip usage of each A/B tile; the sequence loop should avoid loading an overly large K/V block at once.",
        "**用 Autotune 管理候选 tile**：为常见 shape 准备多组 `BLOCK_M/N/K` 和 `multibuffer` 配置，让运行时选择最优组合。": "5. **Use Autotune to manage candidate tiles**: Prepare multiple `BLOCK_M/N/K` and `multibuffer` configurations for common shapes, letting the runtime select the optimal combination.",
        "复杂 Cube 算子的常见风险是把 GPU 上的大量 program 直接迁移到 NPU。若输出 tile 数远大于物理 Cube Core 数，可考虑让每个 program 通过内层循环处理多个 tile，或者在确认逻辑核相互独立时设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 降低调度开销。": "A common migration risk is directly keeping a GPU-style large grid. If the output tile count is far larger than the physical Cube Core count, let each program process multiple tiles in an inner loop, or set `TRITON_ALL_BLOCKS_PARALLEL=1` when logical programs are independent.",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"cube_operator.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process programming_guide/cv_fusion_operator.po
# ============================================================
def fix_cv_fusion_operator_po():
    filepath = os.path.join(BASE_DIR, 'programming_guide', 'cv_fusion_operator.po')
    entries, raw = parse_po(filepath)

    translations = {
        "CV 融合算子开发": "CV Fusion Operator Development",
        "CV 融合算子指同一个算子中同时使用 Cube Core 和 Vector Core：Cube Core 通常负责 `tl.dot`、矩阵乘或卷积式主计算，Vector Core 负责 bias、activation、softmax、归约、mask、layout 重排或跨块同步。CV 融合的目标是减少 kernel 边界和 GM 往返，但需要同时控制 Cube tile、Vector tile、UB/L1 占用和同步关系。": "CV fusion operators use Cube Cores and Vector Cores in the same operator. Cube Cores usually handle `tl.dot`, matrix multiplication, or convolution-like main computation, while Vector Cores handle bias, activation, softmax, reductions, masks, layout reorganization, or cross-block synchronization. The goal is to reduce kernel boundaries and GM round trips while controlling Cube tiles, Vector tiles, UB/L1 usage, and synchronization.",
        "CV 融合简单算子开发": "Simple CV Fusion Operator Development",
        "简单 CV 融合可以从 `third_party/ascend/tutorials/03-matrix-multiplication.py` 中的 matmul + activation 入手，也可以参考 [融合注意力样例](../examples/04_fused_attention_example.md)。最小路径如下：": "For simple CV fusion, start with matmul plus activation in `third_party/ascend/tutorials/03-matrix-multiplication.py` or the [Fused Attention example](../examples/04_fused_attention_example.md).",
        "先实现稳定的 Cube 主计算，例如 `acc = tl.dot(a, b, acc)`。": "1. Implement a stable Cube main computation such as `acc = tl.dot(a, b, acc)`.",
        "在 accumulator 写回前融合轻量 Vector 后处理，例如 bias、scale、activation 或 dtype cast。": "2. Fuse lightweight Vector post-processing before storing the accumulator, such as bias, scale, activation, or dtype cast.",
        "对较大的 accumulator 使用子块切分，避免 Vector 后处理阶段 UB overflow。": "3. Split large accumulators into sub-blocks to avoid UB overflow in the Vector post-processing stage.",
        "如果需要让一个 Cube 输出块拆给多个 Vector 子块处理，可使用 Ascend 扩展中的 `extension.parallel(..., bind_sub_block=True)` 和 `extension.extract_slice`。": "4. If one Cube output block needs to be split across Vector sub-blocks, use `extension.parallel(..., bind_sub_block=True)` and `extension.extract_slice`.",
        "示例结构：": "Example structure:",
        "简单 CV 融合开发时要保持边界清晰：Cube 负责产生较大的二维 accumulator，Vector 负责同一 tile 内的逐元素或小规模归约。若 Vector 部分需要跨多个 Cube tile 共享状态，就需要引入同步、workspace 或拆分 kernel。": "Keep the boundary simple: Cube produces a 2D accumulator, and Vector performs element-wise or small reductions within the same tile. If Vector logic needs state shared across multiple Cube tiles, introduce synchronization, workspace, or split the kernel.",
        "CV 融合复杂算子开发": "Complex CV Fusion Operator Development",
        "复杂 CV 融合可参考 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 中的 best practice：": "Complex CV fusion can reference the best practices in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):",
        "[`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py)：Decode attention 中 QK/PV 使用 Cube，softmax、mask、指数、归一化和离散 KV 访存重排使用 Vector。": "- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py): In decode attention, QK/PV uses Cube while softmax, mask, exponentiation, normalization, and discrete KV memory access reorganization use Vector.",
        "[`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md)：展示融合 cat、slice、conv1d update 时如何用 `insert_slice`、转置和分核优化减少离散访存与 padding 开销。": "- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md): Demonstrates how to use `insert_slice`, transpose, and core allocation optimization to reduce discrete access and padding overhead when fusing cat, slice, and conv1d update.",
        "复杂 CV 融合建议按数据流分层组织：": "Complex CV fusion is recommended to be organized by data flow layers:",
        "**主计算层**：识别哪些步骤必须走 Cube，例如 QK、PV、GEMM、batched matmul。": "1. **Main compute layer**: Identify the steps that must use Cube, such as QK, PV, GEMM, or batched matmul.",
        "**Vector 后处理层**：识别 softmax、activation、mask、scale、normalization、cat/slice、layout transform 等是否能在同一 tile 内完成。": "2. **Vector post-processing layer**: Identify softmax, activation, mask, scale, normalization, cat/slice, and layout transforms that can finish within the same tile.",
        "**访存重排层**：对离散 KV cache、MoE token 重排、短尾轴 tensor，优先在 UB 中用 `insert_slice`、`extract_slice`、转置或借轴转置形成硬件友好的连续访问。": "3. **Memory reorganization layer**: For discrete KV cache, MoE token reordering, or short tail-axis tensors, use `insert_slice`, `extract_slice`, transpose, or axis borrowing in UB to form hardware-friendly continuous access.",
        "**流水和同步层**：通过 `multibuffer`、`set_workspace_multibuffer`、`tile_mix_vector_loop`、`tile_mix_cube_loop` 等编译选项探索 Cube 与 Vector 的重叠执行。": "4. **Pipeline and synchronization layer**: Explore Cube/Vector overlap with options such as `multibuffer`, `set_workspace_multibuffer`, `tile_mix_vector_loop`, and `tile_mix_cube_loop`.",
        "**分核层**：CV 融合算子通常按 Cube Core 数量发射 grid；运行时会以约 1:2 的比例协同 Vector Core。不要简单沿用 GPU 上的大 grid。": "5. **Core allocation layer**: CV fusion operators usually launch by Cube Core count, while Vector Cores cooperate at roughly a 1:2 ratio. Do not directly reuse large GPU grids.",
        "对于 attention 类 CV 融合，推荐先让非 causal、短序列、小 head_dim 的 case 跑通，再逐步加入：": "For attention-style CV fusion, start with non-causal, short-sequence, small-head-dimension cases, and then gradually add:",
        "causal mask 分阶段处理。": "- Causal mask processing in phases.",
        "长序列 K/V block 循环。": "- Long-sequence K/V block loops.",
        "`m_i`/`l_i` 的数值稳定 softmax 更新。": "- Numerically stable `m_i`/`l_i` softmax updates.",
        "HEAD_DIM 较大时的 accumulator workspace 和子块切分。": "- Accumulator workspace and sub-block splitting for large `HEAD_DIM`.",
        "KV cache 离散索引下的 load 重排。": "- Load reorganization for discrete KV-cache indices.",
        "复杂 CV 融合调优时，优先观察 profiling 中 Cube、Vector、MTE2 的时间占比。如果 Cube 等待 Vector，考虑减少 Vector 后处理粒度或打开 CV balance 相关选项；如果 Vector 等待搬运，优先检查离散访存、tail-axis padding 和 multibuffer 配置。": "When tuning complex CV fusion, inspect the Cube, Vector, and MTE2 time ratios in profiling. If Cube waits for Vector, reduce the Vector post-processing granularity or enable CV balance options. If Vector waits for data movement, check irregular access, tail-axis padding, and multibuffer settings first.",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"cv_fusion_operator.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process migration_guide/architecture_difference.po
# ============================================================
def fix_architecture_difference_po():
    filepath = os.path.join(BASE_DIR, 'migration_guide', 'architecture_difference.po')
    entries, raw = parse_po(filepath)

    translations = {
        "昇腾与GPU的开发差异": "Development Differences Between Ascend and GPUs",
        "多核任务并行策略": "Multi-Core Task Parallelism Strategy",
        "NPU在Triton多核并行中是物理核强绑定模式，与GPU逻辑维度并行+硬件自动物理映射的模式形成核心差异": "NPUs are strongly bound to physical cores in Triton multi-core parallelism. This represents a core difference from GPUs' logical dimension parallelism + automatic physical mapping in hardware.",
        "核心对比": "- Core comparison",
        "维度": "Dimension",
        "GPU（NVIDIA）": "GPU (NVIDIA)",
        "昇腾（Ascend）": "Ascend",
        "grid 本质": "Essence of grids",
        "逻辑任务维度（和物理核解耦）": "Logical task dimension (decoupled from physical cores)",
        "物理核组映射（绑定 AI Core 拓扑）": "Physical core group mapping (bound to the AI core topology)",
        "核数 / 维度限制": "Limit on the number of cores/dimensions",
        "grid 维度 / 大小无硬限制": "No hard limit on the grid dimensions/sizes",
        "grid 大小≤AI Core 总数，2D 需匹配拓扑": "Grid size ≤ Total number of AI cores; topology matching required by 2D",
        "GPU:可绑定多个维度轴（三维grid=[n,m,l] 等同于乘积n×m×l个并行线程），每个线程仅对应一次kernel执行，且仅执行一次。 NPU:Vector核，Cube核属于多个物理核，不同代际硬件核数不同，每个核仅执行一次Block,且支持对该Block重复调度执行。": "GPUs can be bound to multiple dimensions (a 3D grid of `[n, m, l]` is equivalent to `n x m x l` parallel threads). Each thread corresponds to only one kernel execution and executes only once. In NPUs, vector cores and cube cores belong to multiple physical cores. The number of cores varies with the generation of hardware. Each core executes only one block and can schedule the block execution repeatedly.",
        "充分利用核数": "Full Utilization of Cores",
        "昇腾NPU具备多个计算核心，合理分配并充分利用所有可用核心，是提升算子性能的关键因素之一。 在调用Triton内核函数时，通过设置launch参数控制使用的核数量。以GELU算子为例：": "Ascend NPUs have multiple computing cores. Properly allocating and fully utilizing all available cores is one of the key factors to improve operator performance. When calling Triton kernel functions, you can set the **launch** parameter to control the number of cores in use. Take the GELU operator as an example:",
        "通过对核数的调优，可实现对所有计算资源的充分调度和利用，从而最大化并行度与吞吐量。注意，当前版本核数需小于等于65535。": "By optimizing the number of cores, you can fully schedule and utilize all computing resources, thereby maximizing the degree of parallelism (DOP) and throughput. Note that the number of cores in the current version must be less than or equal to 65,535.",
        "单核数据搬运策略": "Single-Core Data Transfer Strategy",
        "数据切分Tiling": "Data Tiling",
        "写Triton内核函数时，合理的数据切分策略对性能优化至关重要。通过调整不同的切分粒度参数，可以在不同维度上平衡计算负载与内存访问效率。": "When you write Triton kernel functions, a proper data tiling strategy is essential for performance optimization. By adjusting tiling granularity parameters, you can balance computational workload and memory access efficiency across different dimensions.",
        "常见的切分参数包括：": "Common tiling parameters include:",
        "开发者可根据实际场景手动选择最优的切分配置，使得每次计算尽可能充分利用片上内存（On-chip Memory），避免频繁访问全局内存（Global Memory）造成的 性能瓶颈。": "By manually selecting the optimal tiling configurations based on your actual scenario, you can maximize the utilization of on-chip memory during each computation cycle, preventing performance bottlenecks caused by frequent access to the global memory.",
        "以GELU算子为例，通过调整切分参数，可以有效适配片上缓存容量限制，从而提升执行效率。": "Taking the GELU operator as an example, adjusting the tiling parameters helps effectively adapt to the on-chip cache capacity limit, thereby improving execution efficiency.",
        "注：Atlas 800T/I A2产品的片上内存容量为192KB，因此在设计切分策略时需考虑该限制，确保每轮计算的数据量不超过片上内存容量。": "Note: Atlas 800T/I A2 has an on-chip memory capacity of 192 KB. When designing the tiling strategy, ensure that the data volume of each computation cycle does not exceed this capacity.",
        "GELU算子示例": "Example GELU Operator",
        "GELU算子开发示例，使用3种方式计算结果。": "The following demonstrates the development of an example GELU operator with three result computation methods.",
        "standard_unary      为标准Torch计算。": "`standard_unary` is standard Torch computation.",
        "triton_easy_kernel  为简单Triton实现。": "`triton_easy_kernel` is a simple implementation of Triton.",
        "triton_better_kernel为更高效的Triton实现。": "`triton_better_kernel` is a more efficient implementation of Triton.",
        "标准Torch写法": "Standard Torch Writing",
        "输入tensor x0，经过torch计算实现 GELU 算子，返回结果值。": "After computing the input `tensor x0`, Torch implements the GELU operator and returns the result value.",
        "简单Triton写法": "Simple Triton Writing",
        "以下是一个使用 Triton 编写的简单内核示例，用于展示如何定义和调用一个基本的Triton内核函数。此示例实现了一个简单的数学运算（GELU 激活函数）。": "The following is an example of a simple kernel written in Triton, demonstrating how to define and call a basic Triton kernel function. This example implements a simple mathematical operation (GELU activation function).",
        "注意事项": "Precautions",
        "内存限制：上述写法中，所有输入数据一次性被加载到内存中进行计算。如果输入张量过大，可能会超出单个内核的片上内存容量，导致内存溢出错误。 因此，这种简单的写法更适合于小规模张量的计算或用于理解 Triton 内核的基本写法和调用方式。": "1. Memory limit: In the preceding writing, all input data is loaded to memory at a time for computation. If the input tensor is too large, it may exceed the on-chip memory capacity of a single kernel, resulting in a memory overflow error. Therefore, this simple writing is suitable for computing small-scale tensors or for understanding the basic writing and call method of Triton kernels.",
        "适用场景：尽管这种方法有助于快速理解和入门 Triton 编程，但对于大规模数据集或高性能要求的应用场景，建议采用更复杂的数据切分策略（如 Tiling）， 以充分利用硬件资源并避免内存溢出问题。通过这种方式，开发者可以快速上手 Triton 编程，同时了解如何定义、调用以及优化 Triton 内核函数。": "2. Application scenarios: This method helps developers quickly understand and get started with Triton programming. However, for large-scale data sets or scenarios demanding high performance, developers are advised to use more complex data tiling strategies to fully utilize hardware resources and prevent memory overflow. In this way, developers can quickly get started with Triton programming and understand how to define, call, and optimize Triton kernel functions.",
        "更高效triton写法": "More Efficient Triton Writing",
        "在昇腾 NPU 上使用 Triton 编写高性能算子时，为了充分利用硬件资源、避免内存溢出并提升执行效率，通常需要采用数据切分（Tiling）策略。 下面是一个经过优化的 Triton 内核实现示例，适用于大规模张量计算。": "When using Triton to write high-performance operators on Ascend NPUs, developers need to use a data tiling strategy to fully utilize hardware resources, prevent memory overflow, and improve execution efficiency. The following is an example of an optimized Triton kernel implementation suitable for large-scale tensor computation.",
        "关键代码解释": "Explanation of key code:",
        "编译优化能力": "Compilation Optimization",
        "AscendNPU IR优化": "Ascend NPU IR Optimization",
        "针对昇腾软硬件特性，适配了AscendNPU IR优化的编译选项，如下表所示。 **使用方法**：在autotune的配置阶段，传入编译选项的值 以开启`multibuffer`选项举例，在autotune的配置阶段，即`triton.Config`中，传入`'multibuffer': True`，详见[autotune示例](../examples/06_autotune_example.md)：": "The following table lists the compilation options for Ascend NPU IR optimization, which are adapted to the hardware and software features of Ascend. **Usage**: During the autotune configuration phase, pass the values of the compilation options. For example, to enable the `multibuffer` option, pass `'multibuffer': True` to `triton.Config` during the autotune configuration phase. For details, see [Autotune Example](../examples/06_autotune_example.md).",
        "选项": "Option",
        "能力": "Capability",
        "是否开启": "Enabled or Not",
        "multibuffer": "multibuffer",
        "开启流水并行数据搬运": "Data transfer through parallel pipelines.",
        "默认true； true , false。 autotune中可配置": "Default: **true**. Options: **true** and **false**. It is configurable during autotune.",
        "unit_flag": "unit_flag",
        "cube搬出的一个优化项": "Optimization item for cube-out.",
        "默认None；true , false。  autotune中可配置": "Default: None. Options: **true** and **false**. It is configurable during autotune.",
        "limit_auto_multi_buffer_only_for_local_buffer": "limit_auto_multi_buffer_only_for_local_buffer",
        "CV算子一个优化项，cube搬出的一个优化项": "Optimization item for CV operators and cube-out.",
        "默认None；true , false。 autotune中可配置": "Default: None. Options: **true** and **false**. It is configurable during autotune.",
        "limit_auto_multi_buffer_of_local_buffer": "limit_auto_multi_buffer_of_local_buffer",
        "cube算子开启double buffer具体的scope": "Scope of enabling double buffer for cube operators.",
        '默认None；["no-limit", "no-l0c"]， autotune中可配置': 'Default: None. Value range: ["no-limit", "no-l0c"]. It is configurable during autotune.',
        "set_workspace_multibuffer": "set_workspace_multibuffer",
        "只有在limit_auto_multi_buffer_only_for_local_buffer=false场景下生效": "It takes effect only when **limit_auto_multi_buffer_only_for_local_buffer** is set to **false**.",
        "默认None；如[2,4]，autotune中可配置": "Default: None. Example: [2,4]. It is configurable during autotune.",
        "enable_hivm_auto_cv_balance": "enable_hivm_auto_cv_balance",
        "set_workspace_multibuffer只有在limit_auto_multi_buffer_only_for_local_buffer=false场景下生效": "**set_workspace_multibuffer** takes effect only when **limit_auto_multi_buffer_only_for_local_buffer** is set to **false**.",
        "tile_mix_vector_loop": "tile_mix_vector_loop",
        "CV算子的一个优化项，当前vector可以切几份": "Optimization item for CV operators. It specifies the number of segments into which the current vector can be split.",
        "默认None；如 [2,4,8]，autotune中可配置": "Default: None. Example: [2,4,8]. It is configurable during autotune.",
        "tile_mix_cube_loop": "tile_mix_cube_loop",
        "CV算子一个优化项，当前cube可以切几份": "Optimization item for CV operators. It specifies the number of segments into which the current cube can be split.",
        "auto_blockify_size": "auto_blockify_size",
        "TRITON_ALL_BLOCKS_PARALLEL优化项，用于指定扩展的左起第一个维度的大小。": "Optimization item for TRITON_ALL_BLOCKS_PARALLEL, used to specify the size of the first dimension from the left for expansion.",
        "默认1；如 [2,4,8]，autotune中可配置": "Default: 1. Example: [2,4,8]. It is configurable during autotune.",
        "注：优化编译选项在ascend/backend/compiler.py代码中。": "- Note: The compilation optimization options are located in **ascend/backend/compiler.py**.",
        "注：CV算子表示该算子运算过程中既使用了AI Core又使用了Vector Core。": "- Note: CV operators indicate that both AI cores and vector cores are used during operator computation.",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"architecture_difference.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process migration_guide/migrate_from_gpu.po
# ============================================================
def fix_migrate_from_gpu_po():
    filepath = os.path.join(BASE_DIR, 'migration_guide', 'migrate_from_gpu.po')
    entries, raw = parse_po(filepath)

    translations = {
        "GPU Triton算子迁移": "Migrating Triton Operators from GPUs",
        "概述：本文介绍 GPU Triton 算子迁移到昇腾 NPU 时的通用处理思路和常见问题。迁移时建议先完成 Python 侧设备与运行时接口替换，再检查 grid 分核、访存对齐、单核计算、UB 空间和 coreDim 限制，最后结合具体示例完成代码修改和正确性验证。": "This document describes the general procedure and common issues for migrating GPU Triton operators to Ascend NPUs. Start by replacing Python-side device and runtime interfaces, then check grid core allocation, memory access alignment, single-program computation, UB usage, and coreDim limits. The examples later in this document show how to apply these steps in code.",
        "通用迁移流程": "General Migration Procedure",
        "迁移 Python 侧设备和运行时接口": "Migrate Python-Side Device and Runtime Interfaces",
        "在修改具体 Triton kernel 前，先完成 Python 侧设备迁移：": "Before modifying a specific Triton kernel, migrate the Python-side device code first:",
        "在 Python 文件中增加 `import torch_npu`。": "1. Add `import torch_npu` to the Python file.",
        '查找 `device="cuda"`、`device=\'cuda\'`、`.cuda()` 和 `.to("cuda")` 等设备指定方式，改为 `device="npu"`、`device=\'npu\'`、`.npu()` 或 `.to("npu")`。': '2. Find `device="cuda"`, `device=\'cuda\'`, `.cuda()`, `.to("cuda")`, and similar device specifications, and change them to `device="npu"`, `device=\'npu\'`, `.npu()`, or `.to("npu")`.',
        "查找 `torch.cuda.*`、CUDA stream、CUDA event、CUDA synchronize 等 GPU 专属接口，改为 NPU 对应接口或删除不必要的同步逻辑。": "3. Find GPU-specific APIs such as `torch.cuda.*`, CUDA streams, CUDA events, and CUDA synchronization, then replace them with NPU counterparts or remove unnecessary synchronization logic.",
        "删除只为 GPU 设备发现服务的逻辑，例如 `triton.runtime.driver.active.get_active_torch_device()` 相关设备断言。": "4. Remove logic that exists only for GPU device discovery, such as assertions around `triton.runtime.driver.active.get_active_torch_device()`.",
        "保持 Triton kernel 主体逻辑不变，先用 NPU Tensor 完成编译和正确性验证。": "5. Keep the Triton kernel body unchanged at first, and use NPU tensors to verify compilation and correctness.",
        "调整 grid 分核": "Adjust Grid Core Allocation",
        "GPU 上常见的写法会把 grid 设计为大量逻辑 program，由硬件和运行时调度到 SM 上执行。迁移到 NPU 时，应优先考虑物理 AI Core 数量和算子类型：": "GPU kernels often use a large logical grid and rely on the runtime and hardware to schedule programs onto SMs. On NPUs, first consider the physical AI Core count and operator type:",
        "grid 优先使用 1D；2D NPU 适配写法也会合并为 1D，例如 `(20,)` 与 `(4, 5)` 的效果相同。": "- Prefer 1D grids. NPU 2D adaptations are merged into 1D; for example, `(20,)` and `(4, 5)` produce equivalent execution results.",
        "Vector-only 算子的并发任务数通常按 Vector Core 数量组织；包含 `tl.dot` 的算子通常按 AI Core 数量组织。": "- For Vector-only operators, organize concurrent tasks around the Vector Core count. For operators containing `tl.dot`, organize concurrent tasks around the AI Core count.",
        "当逻辑 grid 远大于物理核数时，需要评估是否改成每个 program 内部循环处理多个 tile，或在逻辑核之间无顺序依赖时使用 `TRITON_ALL_BLOCKS_PARALLEL`。": "- If the logical grid is much larger than the physical core count, consider letting each program process multiple tiles in an inner loop, or use `TRITON_ALL_BLOCKS_PARALLEL` when logical programs have no ordering dependency.",
        "coreDim 不能超过 `UINT16_MAX`（65535），大 shape 算子需要结合 BLOCK_SIZE 或分块方式控制 grid 大小。": "- `coreDim` cannot exceed `UINT16_MAX` (65535). For large shapes, control grid size through BLOCK_SIZE or tiling.",
        "维度": "Dimension",
        "核心结构": "Core Structure",
        "算子类型": "Operator Type",
        "昇腾 NPU (Ascend)": "Ascend NPU",
        "多个 AI Core，分为 Cube Core（矩阵乘）和 Vector Core（向量计算）": "Multiple AI cores, categorized into Cube Cores for matrix multiplication and Vector Cores for vector computation",
        "Vector-only 算子 → 并发任务数 = Vector Core 数；含 `tl.dot` 算子 → 并发任务数 = AI Core 数": "Vector-only operators -> concurrent task count = Vector Core count; operators containing `tl.dot` -> concurrent task count = AI Core count",
        "GPU NVIDIA/AMD": "NVIDIA/AMD GPU",
        "多个 CUDA Core（标量/向量计算） + Tensor Core（矩阵乘）": "Multiple CUDA cores for scalar/vector computation and Tensor Cores for matrix multiplication",
        "GPU 算子一般由编译器和硬件自动决定并发度": "GPU concurrency is generally determined by the compiler and hardware",
        "检查单核数据搬运": "Check Single-Program Data Transfer",
        "完成设备替换后，需要继续检查单个 program 内的数据搬运方式：": "After device replacement, check data movement inside each program:",
        "Vector 算子场景下要求 32 字节访存对齐，cube-vector 融合算子场景下要求 512 字节对齐。": "- Vector operators require 32-byte memory access alignment, and cube-vector fused operators require 512-byte alignment.",
        "保留 tail mask，确认边界元素不会越界访问。": "- Keep tail masks and verify that boundary elements are not accessed out of bounds.",
        "检查一次 tile 的片上内存占用，避免触发 UB 空间溢出。": "- Check on-chip memory usage for each tile to avoid UB overflow.",
        "移除或替换 GPU 专属同步 API，例如 CUDA thread、stream、event 或 kernel synchronize 相关接口。": "- Remove or replace GPU-specific synchronization APIs, such as CUDA thread, stream, event, or kernel synchronization interfaces.",
        "检查单核数据运算": "Check Single-Program Computation",
        "NPU 与 GPU 的计算单元和支持的数据类型存在差异。迁移后应先保证正确性，再根据性能问题调整：": "NPU and GPU compute units differ in supported data types and execution behavior. After migration, verify correctness first, then adjust based on performance symptoms:",
        "对整数索引、offset、长度等中间值，优先确认当前数据类型是否被 NPU 路径高效支持。": "- For integer indices, offsets, and lengths, confirm whether the current dtype is efficiently supported by the NPU path.",
        "对含 `tl.dot` 的算子，确认 M/N/K tile、累加 dtype 和输出 dtype 是否符合 NPU 后端要求。": "- For operators containing `tl.dot`, check M/N/K tiling, accumulator dtype, and output dtype.",
        "对长序列、长 hidden size 或大 K 维循环，优先通过 tiling 控制单次搬入和计算规模。": "- For long sequence, long hidden size, or large K loops, use tiling to control the amount of data moved and computed at one time.",
        "迁移示例": "Migration Examples",
        "示例 1：向量加法完整迁移": "Example 1: Complete Vector Addition Migration",
        "示例 2：设备替换与单核数据搬运": "Example 2: Device Replacement and Single-Program Data Transfer",
        "以下示例展示将设备从 CUDA 替换为 NPU 后，对单核数据搬运场景进行正确性验证：": "The following example replaces CUDA tensors with NPU tensors and verifies correctness for a single-program data transfer case.",
        "常见问题概览": "FAQ",
        "完成迁移基础步骤后，可能会遇到新的问题，新问题可归纳为以下两类：   1.coreDim限制问题   当网格维度超过NPU硬件限制时触发。   典型错误信息：coreDim=xxxx can't be greater than UINT16_MAX   2.UB空间溢出   内存使用超出NPU缓存容量。   典型错误信息：ub overflow, requires xxxx bits while 1572684 bits available!": "After completing the basic migration procedure, you may encounter the following two types of new issues:",
        "解决 coreDim 超限问题": "Solving the coreDim Limit Issue",
        "问题分析:    NPU的 coreDim 参数不能超过 UINT16_MAX（65535）。当处理大规模数据时，简单的grid划分可能导致该限制被突破。": "Issue analysis: The **coreDim** parameter of NPUs cannot exceed **UINT16_MAX** (**65535**). When processing large-scale data, simplistic grid division may exceed this limit.",
        "案例：zeros_like 函数优化     数据规模：N = 1073741824，原始 BLOCK_SIZE = 2048，计算得到的 coreDim = 524288 > 65535（超限）": "Case: Optimizing the `zeros_like` function (data scale `N = 1073741824`; original `BLOCK_SIZE = 2048`; calculated `coreDim = 524288`, exceeding the limit of **65535**).",
        "优化前的代码：": "Code before optimization:",
        "优化后的代码：": "Code after optimization:",
        "动态计算适合的 BLOCK_SIZE 以避免 coreDim 超限": "Dynamically Calculating **BLOCK_SIZE** to Ensure **coreDim** Remains Within the Limit",
        "处理复合问题：coreDim + UB 溢出": "Handling the Compound Issue: coreDim + UB Overflow",
        "问题分析:   在某些情况下，解决了 coreDim 问题后可能引发新的UB溢出问题。这通常发生在增大 BLOCK_SIZE 后，单个线程块需要处理的数据量超出了NPU的UB缓存容量。": "Issue analysis: In some scenarios, solving the **coreDim** limit issue may inadvertently trigger a new issue -- UB overflow. This typically occurs when increasing **BLOCK_SIZE** causes the data volume processed by a single thread block to exceed the UB cache capacity of NPUs.",
        "案例：     数据规模：N = 1073741824，原始 BLOCK_SIZE = 4096，计算得到的 coreDim = 262144 > 65535（超限），调整为 BLOCK_SIZE = 32768 后，coreDim = 32768（合规），但出现 UB 溢出": "Case: Data scale `N = 1073741824`; original `BLOCK_SIZE = 4096`; calculated `coreDim = 262144`, exceeding the limit of **65535**. After **BLOCK_SIZE** is adjusted to **32768**, **coreDim** is **32768** (within the limit), but UB overflow occurs.",
        "优化后代码：": "Code after optimization:",
        "为什么会出现UBSIZE超出内存的错误": "Why Does the UBSIZE Out of Memory Error Occur?",
        "离散访存代码逐行对比观察scalar低效映射": "Discrete Memory Access and Inefficient Scalar Mapping Observed by Line-by-Line Code Comparison",
        "设置环境变量TRITON_DEBUG=1, 保存~/.triton/cache/xxx.ttadapter，然后执行": 'Set the environment variable *TRITON_DEBUG* to **1**, save **~/.triton/cache/xxx.ttadapter**, and execute:',
        "优化思路": "Optimization Approach",
        "比如：": "Example:",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"migrate_from_gpu.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process migration_guide/performance_guidelines.po
# ============================================================
def fix_performance_guidelines_po():
    filepath = os.path.join(BASE_DIR, 'migration_guide', 'performance_guidelines.po')
    entries, raw = parse_po(filepath)

    translations = {
        "NPU高性能编程指南": "NPU High-Performance Programming Guide",
        "合并Grid分核": "Combining Grid Cores",
        "一、自动合并Grid分核优化原则": "I. Principles for Automatically Combining Grid Cores",
        "部分场景下，Triton算子从GPU迁移到NPU。由于体系结构的差异，基于GPU开发的Triton算子Grid分核数较多。在NPU上执行时，无法一次全部调度，多轮下发导致下发时延过大，影响算子性能。基于NPU优Triton算子过程中，需要首先检查Grid分核数。当分核数较大时，使用TRITON_ALL_BLOCKS_PARALLEL环境变量提升算子执行性能。": "Some scenarios requiring migration of Triton operators from GPUs to NPUs. Due to architectural differences, the Triton operators developed on GPUs often utilize large grid core counts. When executed on NPUs, these operators cannot be scheduled all at once. Delivering them in batches introduces significant latency and degrades performance. To optimize NPU-based Triton operators, you need to check the grid core counts first. In cases with large grid core counts, set the environment variable *TRITON_ALL_BLOCKS_PARALLEL* to improve operator execution performance.",
        "指令并行优化": "Optimizing Instruction Parallelism",
        "一、指令并行优化核心原则": "I. Core Principles of Instruction Parallelism Optimization",
        "Triton算子在NPU上执行时，为了提升性能，NPU底层提供multi buffer、指令并行等并行机制，将“数据搬入/数据计算/数据搬出”并行起来，以此来提升性能；但是某些场景下存在multi buffer无法使能问题，影响并行度，导致算子执行性能降低；在性能优化过程中，存在此类问题时，可以参考以下几点做排查，并按照代码示例优化： 1、数据搬运和计算存在数据依赖，产生同步，必须依赖Vector运算后，才能触发MTE搬运，导致并行度低； 2、算子内，无多个数据加载或者单次执行完成无Tiling切分，该场景下无法使能multi buffer； 3、multi buffer需要额外增加UB空间的使用，计算过程中UB空间不足，无法使能multi buffer；": "When executing Triton operators, NPUs leverage parallel mechanisms such as multi-buffer and instruction parallelism to parallelize data-in, computation, and data-out, thereby enhancing performance. However, in certain scenarios, the multi-buffer mechanism cannot be enabled, which reduces the degree of parallelism (DOP) and degrades operator execution performance. If this issue occurs during performance optimization, consider the following aspects and implement optimizations based on the provided code examples:",
        "二、代码示例": "II. Code Examples",
        "示例1：减少同步，提升并行度": "- Example 1: Reducing synchronization for higher DOP",
        "在算子调优过程中，增加指令并行度是算子调优的重要手段。在如下的tl.load语句中，当N > M时, load加载的数据只能填充部分data指向的tensor内存空间中，剩下未填充的部分，如果用户未指定other值，则GPU默认填充为0，为了减少用户迁移的适配工作，NPU保持行为和GPU一致。NPU会先用Vector核对data指向的全部内存空间设置为指定值(如果用户未指定other值，同样设置为0)，然后在使用MTE2指令搬运数据到data指向的部分内存空间，这样就会导致MTE2和Vector产生依赖，无法高效并行，影响性能：": "In operator optimization, increasing instruction-level parallelism (DOP) is a critical strategy. In the `tl.load` statement below, when `N` > `M`, the loaded data fills only a portion of the tensor memory space pointed to by `data`. For the remaining unfilled portion, if users do not specify the `other` value, GPUs default to zero-padding. To reduce the adaptation workload of migration, NPUs maintain the same behavior as GPUs. NPUs first use the vector core to set all the memory space pointed to by `data` to a specified value (defaulting to `0` if no `other` value is provided). Subsequently, the MTE2 instruction transfers data to part of the memory space pointed to by `data`. This implementation results in a dependency between MTE2 and vector operations, which limits parallelism and degrades overall performance.",
        "为了提升性能，在load加载数据只能部分填充到指向的内存空间时，如果未填充的部分不影响后续的计算结果，可以在load语句中，添加care_padding=False来去掉默认值的填充，增加并行度，提升性能，上面算子的优化写法如下：": "To increase DOP and enhance performance, when the loaded data fills only a portion of the memory space pointed to by `data`, add `care_padding=False` to the load statement to remove default-value padding, provided that the unfilled portion does not affect subsequent computation results. That is, the preceding operator can be optimized as follows:",
        "示例2：在Triton算子内，使用for循环，增加Tiling，提升并行度": "- Example 2: Using `for` loops in Triton operators to increase tiling and enhance DOP",
        "数据类型优化": "Optimizing Data Types",
        "一、数据类型优化核心原则": "I. Core Principles of Data Type Optimization",
        "A2/A3 向量运算单元的部分运算操作不支持某些数据类型，这种场景下，对应的向量运算会退化为标量运算，影响性能。在确定不影响整体算子精度的情况下，建议使用支持的数据类型，提升性能。 主要涉及以下操作": "Some operations of the A2/A3 vector units do not support certain data types. In this case, the corresponding vector operations will degrade to scalar operations, affecting performance. If the overall operator accuracy is not affected, it is advisable to use supported data types to improve performance. The following operations are involved.",
        "**OP名称**": "**Operator Name**",
        "**不支持的数据类型**": "**Unsupported Data Type**",
        "Vector ADD": "Vector Add",
        "int64": "int64",
        "Vector CMP": "Vector Cmp",
        "int64/int32": "int64/int32",
        "Vector Add Triton算子示例代码": "- Example code of the Triton operator Vector Add",
        "Vector Cmp Triton算子示例代码": "- Example code of the Triton operator Vector Cmp",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"performance_guidelines.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process debug_guide/debugging.po
# ============================================================
def fix_debugging_po():
    filepath = os.path.join(BASE_DIR, 'debug_guide', 'debugging.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Triton-Ascend 调试指南": "Triton-Ascend Debugging Guide",
        "1 引言": "1 Overview",
        "本文档为 **Triton-Ascend 调试指南**，面向参与 Triton 与昇腾（Ascend）NPU 适配开发的工程师，系统性地介绍在 Triton-Ascend 编译与运行过程中常用的调试方法与工具。": "This document is the **Triton-Ascend Debugging Guide**, which is intended for engineers who participate in adapting Triton to Ascend NPU. It systematically describes the common debugging methods and tools used during Triton-Ascend compilation and running.",
        "全文内容概览如下：": "The contents of this document are as follows:",
        "章节": "Section",
        "主要内容": "Description",
        "**1. 概述**": "**1. Overview**",
        "说明调试的核心目标（聚焦 `ttir.mlir` → `ttadapter.mlir` 转换），并对常见问题进行分类指引。": "Describes the core objectives of debugging (focusing on the `ttir.mlir` to `ttadapter.mlir` conversion) and provides guidance on common issues.",
        "**2. 编译流程概览**": "**2. Compilation Process Overview**",
        "介绍 Triton-Ascend 端到端编译链的关键阶段，为后续调试提供上下文基础。": "Describes the key phases of the Triton-Ascend end-to-end compilation chain, providing a context basis for subsequent debugging.",
        "**3. 临时文件指引**": "**3. Temporary File Guide**",
        "详解编译过程中生成的中间文件（如 `.mlir`、`.ll`、`.o` 等）的存储位置与用途，便于人工检查。": "Describes the storage locations and functions of intermediate files (such as the `.mlir`, `.ll`, and `.o` files) generated during the compilation, facilitating manual check.",
        "**4. 解释器模式**": "**4. Interpreter Mode**",
        "介绍如何通过 `TRITON_INTERPRET=1` 在 CPU 上运行 kernel，作为 NPU 计算结果的精度基准。": 'Describes how to set `TRITON_INTERPRET` to `1` to run the kernel on the CPU and use the result as the accuracy benchmark of the NPU computing result.',
        "**5. 调试方法**": "**5. Debugging Methods**",
        "提供多种实用调试手段：<br>• 静态/运行时打印<br>• 编译错误调试方法<br>": "The following practical debugging methods are provided:<br>Static/Runtime printing<br>Compilation error debugging<br>",
        "**附录 A**": "**Appendix A**",
        "常用环境变量速查表，提升调试效率。": "Provides a quick reference table of common environment variables to improve debugging efficiency.",
        "建议开发者结合具体问题，按需查阅对应章节，以高效定位并解决 Triton-Ascend 集成中的各类异常。": "You are advised to refer to the corresponding sections as required to efficiently locate and resolve various exceptions in Triton-Ascend integration.",
        "1.1 Triton-Ascend 常见问题分类与调试指引": "1.1 Triton-Ascend Common Issue Classification and Debugging Guide",
        "在开发过程中，问题通常可归纳为以下几类。下表提供了快速的问题类型辨识与首选调试方法指引。": "During development, issues can be classified into different types. The following table provides guidance for quickly identifying issue types and preferred debugging methods.",
        "问题类型": "Issue Type",
        "典型表现/描述": "Typical Symptom/Description",
        "推荐的首要调试方法": "Preferred Debugging Method",
        "**精度问题**": "**Accuracy issue**",
        "NPU运行结果与标杆参考结果（如PyTorch或Triton CPU解释器）存在差异。": "The NPU running result is different from the benchmark reference result (such as the PyTorch or Triton CPU interpreter).",
        "4. 解释器模式 <br> 5.1 打印调试方法": "4. Interpreter mode<br>5.1 Debugging by printing",
        "**编译错误 (MLIRCompileError)**": "**Compilation error (MLIRCompileError)**",
        "在编译转换阶段失败，通常在Python端抛出 `MLIRCompileError`。": "If the compilation fails in the conversion phase, `MLIRCompileError` is thrown on the Python side.",
        "5.2 编译错误调试方法": "5.2 Compilation error debugging",
        "2 Triton-Ascend 编译流程概览": "2 Triton-Ascend Compilation Process Overview",
        "理解完整的编译链是进行有效调试的基础。Triton-Ascend 的编译过程遵循以下主要阶段：": "Understanding the complete compilation chain is the basis for effective debugging. The compilation process of Triton-Ascend consists of the following phases:",
        "阶段": "Phase",
        "输入": "Input",
        "输出": "Output",
        "工具/组件": "Tool/Component",
        "说明": "Description",
        "**Python Kernel编译**": "**Python Kernel compilation**",
        "`triton_kernel.py` (Python)": "`triton_kernel.py` (Python)",
        "`ttir.mlir` (MLIR)": "`ttir.mlir` (MLIR)",
        "Triton JIT 编译器": "Triton JIT compiler",
        "将用户编写的Triton Python kernel编译为标准Triton IR (TTIR)。": "Compiles the Triton Python kernel written by users into the standard Triton IR (TTIR).",
        "**Triton IR 适配转换**": "**Triton IR adaptation and transformation**",
        "`ttir.mlir`": "`ttir.mlir`",
        "`ttadapter.mlir`": "`ttadapter.mlir`",
        "适配Ascend的Triton后端": "Ascend-adapted Triton backend",
        "**关键调试阶段**。将TTIR转换为面向Ascend NPU后端的适配器IR。": "**Key debugging phase**. Converts TTIR into the adapter IR for the Ascend NPU backend.",
        "**MLIR 编译与代码生成**": "**MLIR compilation and code generation**",
        "`.o` (可执行对象文件)": "`.o` (executable object file)",
        "毕昇编译器 (`bishengir-compile`)": "BiSheng compiler (`bishengir-compile`)",
        "将适配器IR进一步编译并优化，生成可在NPU上执行的二进制代码。": "The adapter IR is further compiled and optimized to generate binary code that can be executed on the NPU.",
        "**本指南的调试重点**集中在第二阶段：`ttir.mlir` → `ttadapter.mlir` 的转换过程，此阶段是 Triton-Ascend 的主要功能。": "**This guide focuses on** the second phase, that is, the `ttir.mlir` to `ttadapter.mlir` conversion. This phase is the main function of Triton-Ascend.",
        "3 Triton-Ascend 临时文件指引": "3 Triton-Ascend Temporary File Guide",
        "在 Triton-Ascend 的编译过程中，系统会生成多种临时文件用于缓存和调试。理解这些文件的位置和用途对于高效调试至关重要。": "During the compilation of Triton-Ascend, the system generates multiple temporary files for caching and debugging. Understanding the location and usage of these files is critical for efficient debugging.",
        "3.1 缓存文件（Cache）": "3.1 Cache",
        "Triton 使用缓存机制来加速重复编译过程。编译过程中生成的中间文件会被缓存在用户目录下，避免重复编译相同的 kernel。": "Triton uses the cache mechanism to accelerate the repeated compilation process. Intermediate files generated during compilation are cached in the user directory to avoid repeated compilation of the same kernel.",
        "缓存目录结构：": "Cache directory structure:",
        "默认路径: ~/.triton/cache/": "- Default path: **~/.triton/cache/**",
        "主要缓存内容:": "Main cache content:",
        "输入文件缓存: 原始 Triton kernel 生成的 ttir.mlir 文件": "- Input file cache: ttir.mlir file generated by the original Triton kernel",
        "输出文件缓存: 经过适配Ascend转换后的 ttadapter.mlir 文件": "- Output file cache: ttadapter.mlir file converted to adapt to Ascend",
        "编译产物缓存: 最终编译生成的可执行文件": "- Compilation product cache: executable file generated after compilation",
        "缓存文件命名约定： 缓存文件通常以 MD5 哈希值命名，确保相同的 kernel 代码对应相同的缓存文件。": "Naming conventions of cache files: Cache files are usually named using MD5 hash values to ensure that the same kernel code corresponds to the same cache file.",
        "**缓存管理建议：**": "**Recommendations for cache management:**",
        "定期清理: 缓存可能占用较多磁盘空间，可定期清理：": "Periodic clearing: Cache files may occupy a large amount of disk space. You can periodically clear the cache files.",
        "调试时禁用缓存: 在调试编译问题时，建议临时禁用缓存以确保每次都重新编译：": "Disabling cache during debugging: You are advised to temporarily disable the cache to ensure that the compilation is performed each time when debugging compilation issues.",
        "缓存验证: 当怀疑缓存导致问题时，可删除相关缓存文件后重新测试。": "Cache verification: If you suspect that the issue is caused by the cache, delete related cache files and perform the test again.",
        "3.2 调试转储文件（Dump Files）": "3.2 Dump Files",
        "通过设置环境变量 TRITON_DEBUG=1，可以在编译过程中将中间表示文件转储到磁盘，这些文件是调试编译问题的关键资源。": "You can set the environment variable **TRITON_DEBUG** to **1** to dump intermediate representation files to disks during compilation. These files are key resources for debugging compilation issues.",
        "转储目录结构：": "Dump directory structure:",
        "默认路径: ~/.triton/dump/": "- Default path: **~/.triton/dump/**",
        "目录命名: 每个编译会话会生成一个以时间戳或唯一ID命名的子目录": "Directory naming: A subdirectory named by a timestamp or unique ID is generated for each compilation session.",
        "主要转储文件:": "Main dump files:",
        "kernel.ttir.mlir: Triton IR 文件（编译输入）": "- kernel.ttir.mlir: Triton IR file (compilation input)",
        "kernel.ttadapter.mlir: 适配器 IR 文件（转换输出）": "- kernel.ttadapter.mlir: adapter IR file (conversion output)",
        "启用调试转储： 即使启用缓存，只要设置了 TRITON_DEBUG=1，系统仍会在每次运行时重新生成转储文件（覆盖同名目录中的文件）。但若缓存命中且跳过编译，则可能不会触发 IR 转换，导致无新 dump 生成。因此调试时建议同时设置：": "Enabling debug dump: Even if the cache is enabled, the system still generates dump files (overriding files in the directory with the same name) each time the system runs as long as **TRITON_DEBUG=1** is set. However, if the cache is hit and compilation is skipped, IR conversion may not be triggered. As a result, no new dump file is generated. Therefore, during debugging, you are advised to set as follows:",
        "3.3 文件生命周期管理": "3.3 File Lifecycle Management",
        "了解这些临时文件的生成时机和清理策略有助于更好地管理调试环境：": "Understanding when these temporary files are generated and how they are cleared helps you better manage the debugging environment.",
        "文件生成时机表：": "File generation time table",
        "文件类型": "File Type",
        "生成阶段": "Generation Phase",
        "触发条件": "Triggering Condition",
        "清理建议": "Clearance Suggestion",
        "缓存文件": "Cache file",
        "每次编译执行时": "During each compilation",
        "缓存未命中时生成": "Generated when the cache is not hit",
        "定期清理或问题排查时清除": "Periodic clearing or clearing during troubleshooting",
        "转储文件": "Dump file",
        "设置 TRITON_DEBUG=1 后": 'After **TRITON_DEBUG=1** is set',
        "每次编译都会生成": "Generated during each compilation",
        "调试结束后手动清理": "Manual clearing after debugging",
        "生产环境中应禁用调试转储（不设置 TRITON_DEBUG=1）": '- In the production environment, debug dump should be disabled (that is, **TRITON_DEBUG=1** is not set).',
        "缓存机制可以显著提升性能，不应轻易禁用": "- The cache mechanism can significantly improve performance and should not be disabled.",
        "通过合理利用这些临时文件，开发者可以更高效地定位和解决 Triton-Ascend 在编译过程中遇到的问题。": "By properly using these temporary files, developers can efficiently locate and solve issues encountered during Triton-Ascend compilation.",
        "3.4 IR文件解析": "3.4 IR File Parsing",
        "3.4.1 TTIR（Triton Intermediate Representation）": "3.4.1 Triton Intermediate Representation (TTIR)",
        "3.4.1 TTAdapter IR（Target-Specific Adapter Representation）": "3.4.1 Target-Specific Adapter Representation (TTAdapter IR)",
        "4 解释器模式": "4 Interpreter Mode",
        "5 调试方法": "5 Debugging Methods",
        "5.1 打印调试方法": "5.1 Debugging by Printing",
        "5.1.1 静态打印调试方法": "5.1.1 Static Printing Debugging",
        "5.1.2 运行时调试方法": "5.1.2 Runtime Debugging",
        "5.1.3 对比两种打印方法": "5.1.3 Comparing the Two Printing Methods",
        "特性": "Feature",
        "`tl.device_print`": "`tl.device_print`",
        "`tl.static_print`": "`tl.static_print`",
        "**执行时机**": "**Execution time**",
        "运行时（kernel 执行时）": "Runtime (kernel execution)",
        "编译时（kernel 编译时）": "Compilation (kernel compilation)",
        "**输出位置**": "**Output location**",
        "运行时标准输出": "Runtime standard output",
        "编译器标准输出": "Compiler standard output",
        "**可打印内容**": "**Print content**",
        "运行时张量值、变量": "Runtime tensor values and variables",
        "编译时常量、常量表达式": "Compilation constants and constant expressions",
        "**性能影响**": "**Impact on performance**",
        "有运行时开销": "There is runtime overhead.",
        "无运行时开销": "No runtime overhead.",
        "**启用环境变量**": "**Enabling environment variables**",
        "`TRITON_DEVICE_PRINT=1`": "`TRITON_DEVICE_PRINT=1`",
        "环境变量说明：": "Description of environment variables:",
        "TRITON_DEVICE_PRINT=1：启用运行时打印，同时也会启用编译时打印": "**TRITON_DEVICE_PRINT=1**: enables runtime printing and compilation printing.",
        "TRITON_DEBUG=1：启用所有调试输出（包括编译时和运行时打印）": "**TRITON_DEBUG=1**: enables all debugging outputs (including compilation and runtime printing).",
        "5.2.1 Python 代码调试方法": "5.2.1 Debugging Python Code",
        "5.2.2 环境变量调试方法": "5.2.2 Debugging Environment Variables",
        "5.2.2.1 `MLIR_ENABLE_DUMP=1`": "5.2.2.1 `MLIR_ENABLE_DUMP=1`",
        "5.2.2.2 `TRITON_ENABLE_LLVM_DEBUG=1`": '5.2.2.2 `TRITON_ENABLE_LLVM_DEBUG=1`',
        "附录 A：常用环境变量速查表": "Appendix A: Quick Reference Table for Common Environment Variables",
        "变量": "Variable",
        "作用": "Description",
        "`TRITON_DEBUG=1`": "`TRITON_DEBUG=1`",
        "启用中间 IR 转储": "Enables intermediate IR dump.",
        "`TRITON_DISABLE_CACHE=1`": "`TRITON_DISABLE_CACHE=1`",
        "禁用编译缓存": "Disables compilation cache.",
        "`TRITON_INTERPRET=1`": "`TRITON_INTERPRET=1`",
        "使用 CPU 解释器执行 kernel": "Uses the CPU interpreter to execute the kernel.",
        "启用运行时打印输出，同时也会启用编译时打印输出": "Enables runtime print output and compilation print output.",
        "`MLIR_ENABLE_DUMP=1`": "`MLIR_ENABLE_DUMP=1`",
        "启用 MLIR 高层 IR 的自动 dump。在每个 MLIR Pass 执行前后，将当前函数的 IR 以可读文本形式输出": "Enables automatic dump of the MLIR high-level IR and outputs the IR of the current function in readable text before and after each MLIR pass is executed.",
        "`TRITON_ENABLE_LLVM_DEBUG=1`": "`TRITON_ENABLE_LLVM_DEBUG=1`",
        "启用 LLVM 后端 CodeGen 阶段的全量调试日志，包括指令选择、寄存器分配、指令调度、机器码生成等底层过程": "Enables full debugging logs in the LLVM backend CodeGen phase, including instruction selection, register allocation, instruction scheduling, and machine code generation.",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"debugging.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process debug_guide/profiling.po
# ============================================================
def fix_profiling_po():
    filepath = os.path.join(BASE_DIR, 'debug_guide', 'profiling.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Triton-Ascend 性能分析方法": "Triton-Ascend Performance Analysis Method",
        "获取性能数据": "Obtaining Performance Data",
        "在进行性能优化之前，需要获取准确的性能数据，了解性能现状，并根据性能现状分析下一步的优化方向。MindStudio提供了多种针对Triton算子性能测试方法，包括上板Profiling、单算子性能仿真流水图等手段。": "Before performance optimization, you need to obtain accurate performance data, understand the current performance status, and analyze the next optimization direction based on the current performance status. MindStudio provides multiple methods for testing the performance of the Triton operator, including board profiling and single-operator performance simulation pipeline.",
        "上板Profiling": "Board Profiling",
        "msProf工具用于采集和分析运行在昇腾AI处理器上算子的关键性能指标，用户可根据输出的性能数据，快速定位算子的软、硬件性能瓶颈，提升算子性能的分析效率。": "The msProf performance analysis tool is used to collect and analyze key performance metrics of operators running on Ascend AI Processors. You can efficiently locate software and hardware performance bottlenecks of operators based on the output performance data, thereby enhancing the overall efficiency of operator performance analysis.",
        "算子仿真流水图": "Operator Simulation Pipeline Diagram",
        "alt text": "alt text",
        "analyse_data_op_summary": "analyse_data_op_summary",
        "analyse_data_waveform": "analyse_data_waveform",
        "analyse_data_code_mapping": "analyse_data_code_mapping",
        "图2 optimization2": "Figure 2 optimization2",
        "以下两个文件中保存了获取的性能数据：": "The following two files save the obtained performance data:",

    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"profiling.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Process libdevice/simt/libdevice_simt_developer_guide.po
# ============================================================
def fix_libdevice_simt_developer_guide_po():
    filepath = os.path.join(BASE_DIR, 'libdevice', 'simt', 'libdevice_simt_developer_guide.po')
    entries, raw = parse_po(filepath)

    translations = {
        "Libdevice 开发者手册": "Libdevice Developer Guide",
        "SIMT 编译示例": "SIMT Compilation Mode Example",
        "使用 SIMT 编译的 triton kernel 示例": "Triton kernel example with SIMT compilation mode",
        "OP概述": "OP Overview",
        "原型:": "Prototype:",
        # Function names (numbered entries)
        "1. triton.language.extra.cann.libdevice.abs": "1. triton.language.extra.cann.abs",
        "3. triton.language.extra.cann.libdevice.acos": "3. triton.language.extra.cann.acos",
        "4. triton.language.extra.cann.libdevice.acosh": "4. triton.language.extra.cann.acosh",
        "5. triton.language.extra.cann.libdevice.add_rd": "5. triton.language.extra.cann.add_rd",
        "6. triton.language.extra.cann.libdevice.add_rn": "6. triton.language.extra.cann.add_rn",
        "7. triton.language.extra.cann.libdevice.add_ru": "7. triton.language.extra.cann.add_ru",
        "8. triton.language.extra.cann.libdevice.add_rz": "8. triton.language.extra.cann.add_rz",
        "9. triton.language.extra.cann.libdevice.asin": "9. triton.language.extra.cann.asin",
        "10. triton.language.extra.cann.libdevice.asinh": "10. triton.language.extra.cann.asinh",
        "11. triton.language.extra.cann.libdevice.atan": "11. triton.language.extra.cann.atan",
        "12. triton.language.extra.cann.libdevice.atan2": "12. triton.language.extra.cann.atan2",
        "13. triton.language.extra.cann.libdevice.atanh": "13. triton.language.extra.cann.atanh",
        "14. triton.language.extra.cann.libdevice.brev": "14. triton.language.extra.cann.brev",
        "15. triton.language.extra.cann.libdevice.byte_perm": "15. triton.language.extra.cann.byte_perm",
        "16. triton.language.extra.cann.libdevice.ceil": "16. triton.language.extra.cann.ceil",
        "17. triton.language.extra.cann.libdevice.clz": "17. triton.language.extra.cann.clz",
        "18. triton.language.extra.cann.libdevice.copysign": "18. triton.language.extra.cann.copysign",
        "19. triton.language.extra.cann.libdevice.cos": "19. triton.language.extra.cann.cos",
        "20. triton.language.extra.cann.libdevice.cosh": "20. triton.language.extra.cann.cosh",
        "21. triton.language.extra.cann.libdevice.cyl_bessel_i0": "21. triton.language.extra.cann.cyl_bessel_i0",
        "22. triton.language.extra.cann.libdevice.div_rd": "22. triton.language.extra.cann.div_rd",
        "23. triton.language.extra.cann.libdevice.div_rn": "23. triton.language.extra.cann.div_rn",
        "24. triton.language.extra.cann.libdevice.div_ru": "24. triton.language.extra.cann.div_ru",
        "25. triton.language.extra.cann.libdevice.div_rz": "25. triton.language.extra.cann.div_rz",
        "26. triton.language.extra.cann.libdevice.erfinv": "26. triton.language.extra.cann.erfinv",
        "27. triton.language.extra.cann.libdevice.exp10": "27. triton.language.extra.cann.exp10",
        "29. triton.language.extra.cann.libdevice.exp2": "29. triton.language.extra.cann.exp2",
        "30. triton.language.extra.cann.libdevice.exp": "30. triton.language.extra.cann.exp",
        "30. triton.language.extra.cann.libdevice.expm1": "30. triton.language.extra.cann.expm1",
        "31. triton.language.extra.cann.libdevice.fast_dividef": "31. triton.language.extra.cann.fast_dividef",
        "32. triton.language.extra.cann.libdevice.fast_expf": "32. triton.language.extra.cann.fast_expf",
        "33. triton.language.extra.cann.libdevice.fdim": "33. triton.language.extra.cann.fdim",
        "34. triton.language.extra.cann.libdevice.ffs": "34. triton.language.extra.cann.ffs",
        "35. triton.language.extra.cann.libdevice.float_as_int": "35. triton.language.extra.cann.float_as_int",
        "36. triton.language.extra.cann.libdevice.floor": "36. triton.language.extra.cann.floor",
        "37. triton.language.extra.cann.libdevice.fma": "37. triton.language.extra.cann.fma",
        "38. triton.language.extra.cann.libdevice.fma_rd": "38. triton.language.extra.cann.fma_rd",
        "39. triton.language.extra.cann.libdevice.fma_rn": "39. triton.language.extra.cann.fma_rn",
        "40. triton.language.extra.cann.libdevice.fma_ru": "40. triton.language.extra.cann.fma_ru",
        "41. triton.language.extra.cann.libdevice.fma_rz": "41. triton.language.extra.cann.fma_rz",
        "42. triton.language.extra.cann.libdevice.fmod": "42. triton.language.extra.cann.fmod",
        "43. triton.language.extra.cann.libdevice.hadd": "43. triton.language.extra.cann.hadd",
        "44. triton.language.extra.cann.libdevice.hypot": "44. triton.language.extra.cann.hypot",
        "45. triton.language.extra.cann.libdevice.lgamma": "45. triton.language.extra.cann.lgamma",
        "46. triton.language.extra.cann.libdevice.log10": "46. triton.language.extra.cann.log10",
        "47. triton.language.extra.cann.libdevice.log2": "47. triton.language.extra.cann.log2",
        "48. triton.language.extra.cann.libdevice.log": "48. triton.language.extra.cann.log",
        "49. triton.language.extra.cann.libdevice.mul24": "49. triton.language.extra.cann.mul24",
        "50. triton.language.extra.cann.libdevice.mul_rd": "50. triton.language.extra.cann.mul_rd",
        "51. triton.language.extra.cann.libdevice.mul_rn": "51. triton.language.extra.cann.mul_rn",
        "52. triton.language.extra.cann.libdevice.mul_ru": "52. triton.language.extra.cann.mul_ru",
        "53. triton.language.extra.cann.libdevice.mul_rz": "53. triton.language.extra.cann.mul_rz",
        "54. triton.language.extra.cann.libdevice.mulhi": "54. triton.language.extra.cann.mulhi",
        "55. triton.language.extra.cann.libdevice.nearbyint": "55. triton.language.extra.cann.nearbyint",
        "56. triton.language.extra.cann.libdevice.nextafter": "56. triton.language.extra.cann.nextafter",
        "57. triton.language.extra.cann.libdevice.popc": "57. triton.language.extra.cann.popc",
        "58. triton.language.extra.cann.libdevice.pow": "58. triton.language.extra.cann.pow",
        "59. triton.language.extra.cann.libdevice.rcp_rd": "59. triton.language.extra.cann.rcp_rd",
        "60. triton.language.extra.cann.libdevice.rcp_rn": "60. triton.language.extra.cann.rcp_rn",
        "61. triton.language.extra.cann.libdevice.rcp_ru": "61. triton.language.extra.cann.rcp_ru",
        "62. triton.language.extra.cann.libdevice.rcp_rz": "62. triton.language.extra.cann.rcp_rz",
        "63. triton.language.extra.cann.libdevice.remainder": "63. triton.language.extra.cann.remainder",
        "64. triton.language.extra.cann.libdevice.rhadd": "64. triton.language.extra.cann.rhadd",
        "65. triton.language.extra.cann.libdevice.rint": "65. triton.language.extra.cann.rint",
        "66. triton.language.extra.cann.libdevice.round": "66. triton.language.extra.cann.round",
        "67. triton.language.extra.cann.libdevice.rsqrt": "67. triton.language.extra.cann.rsqrt",
        "68. triton.language.extra.cann.libdevice.rsqrt_rn": "68. triton.language.extra.cann.rsqrt_rn",
        "69. triton.language.extra.cann.libdevice.sad": "69. triton.language.extra.cann.sad",
        "70. triton.language.extra.cann.libdevice.saturatef": "70. triton.language.extra.cann.saturatef",
        "71. triton.language.extra.cann.libdevice.saturatef": "71. triton.language.extra.cann.signbit",
        "72. triton.language.extra.cann.libdevice.sin": "72. triton.language.extra.cann.sin",
        "72. triton.language.extra.cann.libdevice.sinh": "72. triton.language.extra.cann.sinh",
        "74. triton.language.extra.cann.libdevice.sqrt": "74. triton.language.extra.cann.sqrt",
        "75. triton.language.extra.cann.libdevice.tan": "75. triton.language.extra.cann.tan",
        "75. triton.language.extra.cann.libdevice.tanh": "75. triton.language.extra.cann.tanh",
        "77. triton.language.extra.cann.libdevice.trunc": "77. triton.language.extra.cann.trunc",
        # Function descriptions
        "计算输入参数的绝对值。": "Computes the absolute value of the input parameter.",
        "计算输入参数的反余弦值。": "Computes the inverse cosine (arccos) of the input parameter.",
        "计算输入参数的反双曲余弦值。": "Computes the inverse hyperbolic cosine of the input parameter.",
        "向下舍入浮点数加法。": "Floating-point addition with round-down (toward negative infinity) rounding mode.",
        "最近偶数舍入浮点数加法。": "Floating-point addition with round-to-nearest-even rounding mode.",
        "向上舍入浮点数加法。": "Floating-point addition with round-up (toward positive infinity) rounding mode.",
        "向零舍入浮点数加法。": "Floating-point addition with round-toward-zero rounding mode.",
        "计算输入参数的反正弦值。": "Computes the inverse sine (arcsin) of the input parameter.",
        "计算输入参数的反双曲正弦值。": "Computes the inverse hyperbolic sine of the input parameter.",
        "计算输入参数的反正切值。": "Computes the inverse tangent (arctan) of the input parameter.",
        "反正切函数，计算 x / y 的反正切值。": "Two-argument inverse tangent function, computes the arctangent of x / y.",
        "反双曲正切函数，计算输入参数的反双曲正切值。": "Inverse hyperbolic tangent function, computes the inverse hyperbolic tangent of the input parameter.",
        "位反转函数，反转32位整数的位顺序。": "Bit reversal function, reverses the bit order of a 32-bit integer.",
        "字节排列操作，从两个32位整数中选择字节组成新整数。输入整数 x 和 y 的字节顺序如下": "Byte permutation operation, selects bytes from two 32-bit integers to form a new integer. The byte order of input integers x and y is as follows:",
        "字节选择参数 s 为32位整数，各比特位与字节选择对应关系如下": "The byte selection parameter s is a 32-bit integer, with each bit group corresponding to byte selection as follows:",
        "向上取整，返回大于或等于 x 的最小整数。": "Ceiling operation, returns the smallest integer greater than or equal to x.",
        "计算32位整数的前导零数量。": "Counts the number of leading zeros in a 32-bit integer.",
        "生成一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。": "Generates a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.",
        "计算输入参数（弧度）的余弦值。": "Computes the cosine of the input parameter (in radians).",
        "计算输入参数的双曲余弦值。": "Computes the hyperbolic cosine of the input parameter.",
        "计算输入参数的修正零阶贝塞尔函数值。": "Computes the modified Bessel function of the first kind of order zero for the input parameter.",
        "向下舍入浮点数除法。": "Floating-point division with round-down (toward negative infinity) rounding mode.",
        "最近偶数舍入浮点数除法。": "Floating-point division with round-to-nearest-even rounding mode.",
        "向上舍入浮点数除法。": "Floating-point division with round-up (toward positive infinity) rounding mode.",
        "向零舍入浮点数除法。": "Floating-point division with round-toward-zero rounding mode.",
        "逆误差函数，找到满足 x = erf(y) 的值 y。": "Inverse error function, finds the value y such that x = erf(y).",
        "以 10 为底的指数函数，计算 10 的 x 次方。": "Base-10 exponential function, computes 10 raised to the power of x.",
        "以 2 为底的指数函数，计算 2 的 x 次方。": "Base-2 exponential function, computes 2 raised to the power of x.",
        "指数函数，计算 e 的 x 次方。": "Exponential function, computes e raised to the power of x.",
        "计算 e 的 x 次方减 1 的结果。": "Computes e raised to the power of x, minus 1.",
        "快速近似除法。": "Fast approximate division.",
        "快速近似指数函数。": "Fast approximate exponential function.",
        "计算 x 与 y 的正差。当 x > y 时，返回 x - y，否则返回 0。": "Computes the positive difference between x and y. When x > y, returns x - y; otherwise returns 0.",
        "查找第一个被置为1的位，返回最低被置为1的位的索引。": "Finds the first bit set to 1, returns the index of the lowest bit set to 1.",
        "将浮点数的比特位重新解释为32位整数。不进行数值转换。": "Reinterprets the bit pattern of a floating-point number as a 32-bit integer. No numeric conversion is performed.",
        "向下取整，返回小于或等于 x 的最大整数。": "Floor operation, returns the largest integer less than or equal to x.",
        "融合乘加，计算 x × y + z。": "Fused multiply-add, computes x * y + z.",
        "向下舍入模式下的融合乘加操作。": "Fused multiply-add operation with round-down rounding mode.",
        "最近偶数舍入模式下的融合乘加操作。": "Fused multiply-add operation with round-to-nearest-even rounding mode.",
        "向上舍入模式下的融合乘加操作。": "Fused multiply-add operation with round-up rounding mode.",
        "向零舍入模式下的融合乘加操作。": "Fused multiply-add operation with round-toward-zero rounding mode.",
        "浮点数取模，计算 x / y 的余数，结果与 x 同号。": "Floating-point modulo, computes the remainder of x / y, with the same sign as x.",
        "计算 x 和 y 的平均值。": "Computes the average of x and y.",
        "计算 x 和 y 之间的欧几里得距离。": "Computes the Euclidean distance between x and y.",
        "计算输入为 x 的伽马函数绝对值的自然对数。": "Computes the natural logarithm of the absolute value of the gamma function for input x.",
        "计算输入为 x 的以 10 为底的对数。": "Computes the base-10 logarithm of input x.",
        "计算输入为 x 的以 2 为底的对数。": "Computes the base-2 logarithm of input x.",
        "计算输入为 x 的以 e 为底的对数。": "Computes the natural (base-e) logarithm of input x.",
        "计算 x 和 y 的低24位乘法结果。": "Computes the lower 24-bit multiplication result of x and y.",
        "向下舍入浮点数乘法。": "Floating-point multiplication with round-down rounding mode.",
        "最近偶数舍入浮点数乘法。": "Floating-point multiplication with round-to-nearest-even rounding mode.",
        "向上舍入浮点数乘法。": "Floating-point multiplication with round-up rounding mode.",
        "向零舍入浮点数乘法。": "Floating-point multiplication with round-toward-zero rounding mode.",
        "计算 x 和 y 的乘法结果的高 32 位。": "Computes the high 32 bits of the multiplication result of x and y.",
        "将 x 转换为最近邻整数。": "Converts x to the nearest integer.",
        "计算从 x 方向朝 y 的下一个可表示浮点数。": "Computes the next representable floating-point number from x toward y.",
        "计算 x 中置位为 1 的数量。": "Counts the number of bits set to 1 in x.",
        "幂函数，计算 x 的 y 次方。": "Power function, computes x raised to the power of y.",
        "向下舍入浮点数倒数运算。": "Floating-point reciprocal with round-down rounding mode.",
        "最近偶数舍入浮点数倒数运算。": "Floating-point reciprocal with round-to-nearest-even rounding mode.",
        "向上舍入浮点数倒数运算。": "Floating-point reciprocal with round-up rounding mode.",
        "向零舍入浮点数倒数运算。": "Floating-point reciprocal with round-toward-zero rounding mode.",
        "计算 x 对 y 的余数，满足 r = x - ny，其中 n 是 x / y 的最近邻整数。": "Computes the remainder of x divided by y, where r = x - ny, and n is the nearest integer to x / y.",
        "计算 x 和 y 平均值的取整结果。": "Computes the rounded average of x and y.",
        "按最近偶数舍入模式计算 x 的最近邻整数。": "Computes the nearest integer to x using round-to-nearest-even rounding mode.",
        "计算 x 的平方根倒数。": "Computes the reciprocal square root of x.",
        "按最近偶数舍入模式计算 x 的平方根倒数。": "Computes the reciprocal square root of x using round-to-nearest-even rounding mode.",
        "计算 |x-y|+z，其中 x 和 y 是有符号整数，z 是无符号整数。": "Computes |x-y|+z, where x and y are signed integers and z is an unsigned integer.",
        "将 x 限制在 \\[+0.0, 1.0] 范围内。": "Clamps x to the range [+0.0, 1.0].",
        "获取 x 的符号位。": "Extracts the sign bit of x.",
        "计算输入参数 x （弧度）的正弦值。": "Computes the sine of the input parameter x (in radians).",
        "计算输入参数 x 的双曲正弦值。": "Computes the hyperbolic sine of input parameter x.",
        "计算 x 的平方根值。": "Computes the square root of x.",
        "计算输入参数 x （弧度）的正切值。": "Computes the tangent of input parameter x (in radians).",
        "计算输入参数 x 的双曲正切值。": "Computes the hyperbolic tangent of input parameter x.",
        "截断取整，向零舍入到最近邻整数。": "Truncation operation, rounds toward zero to the nearest integer.",
        # Return value descriptions
        "返回值: `tl.tensor`, 返回输入参数的绝对值。": "Return Value: `tl.tensor`, containing the absolute value of the input parameter.",
        "返回值: `tl.tensor`, 返回输入参数的反余弦值，取值范围 \\[0, π] 弧度。": "Return Value: `tl.tensor`, containing the inverse cosine of the input parameter, in the range [0, pi] radians.",
        "返回值: `tl.tensor`, 返回输入参数的反双曲余弦值，取值范围 \\[0, +∞]。": "Return Value: `tl.tensor`, containing the inverse hyperbolic cosine of the input parameter, in the range [0, +infinity].",
        "返回值: `tl.tensor`, 返回向下舍入的加法结果。": "Return Value: `tl.tensor`, containing the addition result rounded down.",
        "返回值: `tl.tensor`, 返回最近偶数舍入的加法结果。": "Return Value: `tl.tensor`, containing the addition result rounded to the nearest even number.",
        "返回值: `tl.tensor`, 返回向上舍入的加法结果。": "Return Value: `tl.tensor`, containing the addition result rounded up.",
        "返回值: `tl.tensor`, 返回向零舍入的加法结果。": "Return Value: `tl.tensor`, containing the addition result rounded toward zero.",
        "返回值: `tl.tensor`, 返回输入参数的反正弦值，取值范围 \\[-π/2, π/2] 弧度。": "Return Value: `tl.tensor`, containing the inverse sine of the input parameter, in the range [-pi/2, pi/2] radians.",
        "返回值: `tl.tensor`, 返回输入参数的反双曲正弦值。": "Return Value: `tl.tensor`, containing the inverse hyperbolic sine of the input parameter.",
        "返回值: `tl.tensor`, 返回输入参数的反正切值，取值范围 \\[-π/2, π/2] 弧度。": "Return Value: `tl.tensor`, containing the inverse tangent of the input parameter, in the range [-pi/2, pi/2] radians.",
        "返回值: `tl.tensor`, 返回 x / y 的反正切值，取值范围 \\[-π, π] 弧度。": "Return Value: `tl.tensor`, containing the arctangent of x / y, in the range [-pi, pi] radians.",
        "返回值: `tl.tensor`, 返回输入参数的反双曲正切值，取值范围 \\[-1, 1]。": "Return Value: `tl.tensor`, containing the inverse hyperbolic tangent of the input parameter, in the range [-1, 1].",
        "返回值: `tl.tensor`, 返回位反转后的32位整数。": "Return Value: `tl.tensor`, containing the 32-bit integer with reversed bit order.",
        "返回值: `tl.tensor`, 返回值 return\\[n] := input\\[selector\\[n]]，n 表示输出整数的第 n 个字节。": "Return Value: `tl.tensor`, where return[n] := input[selector[n]], where n represents the n-th byte of the output integer.",
        "返回值: `tl.tensor`, 返回向上取整的结果。": "Return Value: `tl.tensor`, containing the ceiling result.",
        "返回值: `tl.tensor`, 返回输入参数的前导零数量。范围 \\[0, 32]。": "Return Value: `tl.tensor`, containing the number of leading zeros in the input parameter. Range: [0, 32].",
        "返回值: `tl.tensor`, 返回一个浮点数，其绝对值等于 x 的绝对值，符号与 y 相同。": "Return Value: `tl.tensor`, containing a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.",
        "返回值: `tl.tensor`, 返回输入参数的余弦值。": "Return Value: `tl.tensor`, containing the cosine of the input parameter.",
        "返回值: `tl.tensor`, 返回输入参数的双曲余弦值。": "Return Value: `tl.tensor`, containing the hyperbolic cosine of the input parameter.",
        "返回值: `tl.tensor`, 返回输入参数的修正零阶贝塞尔函数值。": "Return Value: `tl.tensor`, containing the modified Bessel function of the first kind of order zero for the input parameter.",
        "返回值: `tl.tensor`, 返回除法结果。": "Return Value: `tl.tensor`, containing the division result.",
        "返回值: `tl.tensor`, 返回输入参数的逆误差函数值。": "Return Value: `tl.tensor`, containing the inverse error function of the input parameter.",
        "返回值: `tl.tensor`, 返回 10 的 x 次方的计算结果。": "Return Value: `tl.tensor`, containing the result of 10 raised to the power of x.",
        "返回值: `tl.tensor`, 返回 2 的 x 次方的计算结果。": "Return Value: `tl.tensor`, containing the result of 2 raised to the power of x.",
        "返回值: `tl.tensor`, 返回 e 的 x 次方的计算结果。": "Return Value: `tl.tensor`, containing the result of e raised to the power of x.",
        "返回值: `tl.tensor`, 返回 e 的 x 次方减 1 的计算结果。": "Return Value: `tl.tensor`, containing the result of e raised to the power of x, minus 1.",
        "返回值: `tl.tensor`, 返回快速近似除法的结果。": "Return Value: `tl.tensor`, containing the result of the fast approximate division.",
        "返回值: `tl.tensor`, 返回快速近似指数函数的结果。": "Return Value: `tl.tensor`, containing the result of the fast approximate exponential function.",
        "返回值: `tl.tensor`, 返回 x 与 y 之间的正差。": "Return Value: `tl.tensor`, containing the positive difference between x and y.",
        "返回值: `tl.tensor`, 返回最低被置为1的位的索引，取值范围 \\[0, 32]。": "Return Value: `tl.tensor`, containing the index of the lowest bit set to 1. Range: [0, 32].",
        "返回值: `tl.tensor`, 返回将浮点数的比特位重新解释为32位整数的结果。": "Return Value: `tl.tensor`, containing the bit pattern of the floating-point number reinterpreted as a 32-bit integer.",
        "返回值: `tl.tensor`, 返回向下取整的结果。": "Return Value: `tl.tensor`, containing the floor result.",
        "返回值: `tl.tensor`, 返回融合乘加的结果。": "Return Value: `tl.tensor`, containing the result of fused multiply-add.",
        "返回值: `tl.tensor`, 返回浮点数取模的结果。": "Return Value: `tl.tensor`, containing the result of the floating-point modulo.",
        "返回值: `tl.tensor`, 返回 x 和 y 的平均值。": "Return Value: `tl.tensor`, containing the average of x and y.",
        "返回值: `tl.tensor`, 返回 x 和 y 之间的欧几里得距离。": "Return Value: `tl.tensor`, containing the Euclidean distance between x and y.",
        "返回值: `tl.tensor`, 返回输入为 x 的伽马函数绝对值的自然对数。": "Return Value: `tl.tensor`, containing the natural logarithm of the absolute value of the gamma function for input x.",
        "返回值: `tl.tensor`, 返回输入为 x 的以 10 为底的对数。": "Return Value: `tl.tensor`, containing the base-10 logarithm of input x.",
        "返回值: `tl.tensor`, 返回输入为 x 的以 2 为底的对数。": "Return Value: `tl.tensor`, containing the base-2 logarithm of input x.",
        "返回值: `tl.tensor`, 返回输入为 x 的以 e 为底的对数。": "Return Value: `tl.tensor`, containing the natural logarithm of input x.",
        "返回值: `tl.tensor`, 返回 x 和 y 的低24位乘法结果。": "Return Value: `tl.tensor`, containing the lower 24-bit multiplication result of x and y.",
        "返回值: `tl.tensor`, 返回浮点数乘法的结果。": "Return Value: `tl.tensor`, containing the floating-point multiplication result.",
        "返回值: `tl.tensor`, 返回 x 和 y 的乘法结果的高 32 位。": "Return Value: `tl.tensor`, containing the high 32 bits of the multiplication result of x and y.",
        "返回值: `tl.tensor`, 返回最近邻整数。": "Return Value: `tl.tensor`, containing the nearest integer to x.",
        "返回值: `tl.tensor`, 返回下一个可表示浮点数。": "Return Value: `tl.tensor`, containing the next representable floating-point number.",
        "返回值: `tl.tensor`, 返回 x 中置位为 1 的数量， 取值范围 \\[0, 32]。": "Return Value: `tl.tensor`, containing the number of bits set to 1 in x. Range: [0, 32].",
        "返回值: `tl.tensor`, 返回 x 的 y 次方。": "Return Value: `tl.tensor`, containing x raised to the power of y.",
        "返回值: `tl.tensor`, 返回 1 / x。": "Return Value: `tl.tensor`, containing 1 / x.",
        "返回值: `tl.tensor`, 返回 x 对 y 的余数。": "Return Value: `tl.tensor`, containing the remainder of x divided by y.",
        "返回值: `tl.tensor`, 返回 x 和 y 平均值的取整结果。": "Return Value: `tl.tensor`, containing the rounded average of x and y.",
        "返回值: `tl.tensor`, 返回 x 的最近邻整数。": "Return Value: `tl.tensor`, containing the nearest integer to x.",
        "返回值: `tl.tensor`, 返回 x 的平方根倒数。": "Return Value: `tl.tensor`, containing the reciprocal square root of x.",
        "返回值: `tl.tensor`, 返回 |x-y|+z。": "Return Value: `tl.tensor`, containing |x-y|+z.",
        "返回值: `tl.tensor`, 返回 x 的饱和值，取值范围 \\[+0.0, 1.0]。": "Return Value: `tl.tensor`, containing the saturated value of x, in the range [+0.0, 1.0].",
        "返回值: `tl.tensor`, 返回 x 的符号位。": "Return Value: `tl.tensor`, containing the sign bit of x.",
        "返回值: `tl.tensor`, 返回输入 x 的正弦值。": "Return Value: `tl.tensor`, containing the sine of input x.",
        "返回值: `tl.tensor`, 返回输入 x 的双曲正弦值。": "Return Value: `tl.tensor`, containing the hyperbolic sine of input x.",
        "返回值: `tl.tensor`, 返回 x 的平方根值。": "Return Value: `tl.tensor`, containing the square root of x.",
        "返回值: `tl.tensor`, 返回输入 x 的正切值。": "Return Value: `tl.tensor`, containing the tangent of input x.",
        "返回值: `tl.tensor`, 返回输入 x 的双曲正切值。": "Return Value: `tl.tensor`, containing the hyperbolic tangent of input x.",
        "返回值: `tl.tensor`, 返回取整结果。": "Return Value: `tl.tensor`, containing the truncation result.",
        # Supported types
        "支持类型：`int32`, `float32`": "Supported Datatypes: `int32`, `float32`",
        "支持类型：`float32`": "Supported Datatypes: `float32`",
        "支持类型：`int32`": "Supported Datatypes: `int32`",
        "支持类型：`float32` |": "Supported Datatypes: `float32`",
    }

    fixed = 0
    for entry in entries:
        msgid = get_entry_text(entry, 'msgid')
        current_msgstr = get_entry_text(entry, 'msgstr')
        if msgid in translations:
            expected = translations[msgid]
            if current_msgstr != expected:
                set_entry_text(entry, 'msgstr', expected)
                fixed += 1

    write_po(filepath, entries)
    print(f"libdevice_simt_developer_guide.po: Fixed {fixed} entries")
    return fixed


# ============================================================
# Main execution
# ============================================================
if __name__ == '__main__':
    total = 0
    total += fix_contributing_po()
    total += fix_governance_po()
    total += fix_release_policy_po()
    total += fix_community_technical_meeting_po()
    total += fix_programming_guide_index_po()
    total += fix_vector_operator_po()
    total += fix_cube_operator_po()
    total += fix_cv_fusion_operator_po()
    total += fix_architecture_difference_po()
    total += fix_migrate_from_gpu_po()
    total += fix_performance_guidelines_po()
    total += fix_debugging_po()
    total += fix_profiling_po()
    total += fix_libdevice_simt_developer_guide_po()
    print(f"\nTotal fixed entries: {total}")
