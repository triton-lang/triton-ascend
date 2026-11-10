# Triton-Ascend 最新动态

本文介绍 Triton-Ascend 官方文档（[https://triton-ascend.readthedocs.io](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)）的最新上线、新增及变更动态，按更新时间倒序组织，便于用户及时了解文档更新信息。

软件版本的发布计划、维护策略及配套版本信息不属于本文范围，请参见[版本发布](./release_note.md)。

> **安全声明**：如果您在使用过程中发现安全问题，请按照[安全声明](./community/SECURITYNOTE_zh.md)中提供的渠道进行反馈，请勿在公开渠道披露。

## 2026年09月

| 序号 | 更新点 | 更新说明 | 相关文档 |
| --- | --- | --- | --- |
| 1 | 新增《do_bench_npu 指南》 | 新增 do_bench_npu 性能基准测试工具使用指南，介绍 NPU 场景下的性能测试方法与结果解读，并接入文档站导航索引。 | [do_bench_npu 指南](./debug_guide/do_bench_npu_guide.md) |
| 2 | 新增"用户自定义 Autotune 反馈"章节 | 《Autotune 指南》新增用户自定义 autotune 反馈机制的说明，支持用户将调优经验沉淀并反馈到编译流程。 | [Autotune 指南](./autotune_guide.md) |
| 3 | 新增"源码构建选项"说明 | 《安装指南》新增源码构建选项（Build Options）说明，指导用户在源码编译时按需配置构建参数。 | [安装指南](./installation_guide.md) |
| 4 | 更新《libdevice 开发指南》 | 大幅修订 libdevice 开发指南，扩充算子约束说明，并新增 `fast_tanhf`、`float2half_rn`、`half2float`、`max`、`min`、`nan` 等算子示例。 | [libdevice 开发指南](./libdevice/libdevice_developer_guide.md) |
| 5 | 更新《快速开始》 | 扩展"下一步开发"索引，帮助用户从快速开始平滑衔接到算子开发、算子迁移、调试调优等进阶文档。 | [快速开始](./quick_start.md) |
| 6 | 更新《环境变量与编译选项》 | 同步编译选项清理（compile options cleanup）变更，更新新旧编译选项的映射关系与迁移说明。 | [环境变量与编译选项](./environment_variable_and_compiler_options_reference.md) |
| 7 | 更新《libdevice API 文档》 | 统一 libdevice 文档示例的精度断言至 1e-4，提升示例规范性与一致性。 | [libdevice API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 8 | 修复扩展 API 文档问题 | 修复 extension 与 bl 文档的渲染问题，并为 `sub_vec_num` 算子新增用法示例。 | [Extension Ops API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.extra.cann.extension.html) |
| 9 | 更新《社区技术例会》 | 更新技术例会日历（icalendar）订阅信息，方便社区成员订阅例会安排。 | [技术例会](./community/community_technical_meeting.md) |
| 10 | 文档质量优化 | 批量修复文档勘误、格式与链接问题，统一产品命名规范（如 Atlas A2/A3 等表述）。 | / |

## 2026年06月

| 序号 | 更新点 | 更新说明 | 相关文档 |
| --- | --- | --- | --- |
| 1 | 新增《libdevice API 文档》 | 新增 libdevice 自动生成（autodoc）文档，覆盖 `acos`、`asin`、`atan`、`fast_dividef`、`fast_expf` 等 60 余个算子的使用约束与可运行示例。 | [libdevice API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 2 | 新增《triton_launch_kernel》API 文档 | 新增 bl 域 `triton_launch_kernel` 扩展 API 的说明与用例。 | [Extension Ops API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.extra.cann.extension.html) |
| 3 | 新增《tl.topk》算子文档 | 新增 `tl.topk` 算子的 API 说明与示例。 | [triton.language API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.html) |
| 4 | 《算子开发指南》新增"多维张量切片"章节 | 补充多维度张量切片（slicing）的使用方法与示例。 | [算子开发指南](./programming_guide/index.md) |
| 5 | 新增访存使用约束说明 | 补充 `tl.load`/`tl.store` 中 `cache_modifier`、`volatile` 参数的支持情况，以及控制流场景下的访存使用约束。 | [triton.language API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.html) |
| 6 | 批量更新算子 API 文档 | 更新数学、规约、访存、创建类、Scan/Sort、Atomic/Rand/Histogram 等算子的约束与示例，覆盖 `gather_out_to_ub`、`index_put`、`scatter_ub_to_out` 等扩展算子。 | [triton.language API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.html) |
| 7 | 更新《安装指南》《快速开始》 | 优化安装流程与快速上手指引内容。 | [安装指南](./installation_guide.md) |
| 8 | 《调试指南》新增 device_print 示例 | 补充 `device_print` 的运行时输出示例。 | [算子调试指南](./debug_guide/debugging.md) |
| 9 | 《FAQ》新增 UB Overflow 处理说明 | 新增 UB Overflow 问题的排查处理说明及 Triton 使用约束条目。 | [FAQ](./FAQ.md) |
| 10 | 《迁移指南》新增 auto-blockify 机制说明 | 在架构差异章节补充 auto-blockify 自动分块机制的迁移说明。 | [架构差异说明](./migration_guide/architecture_difference.md) |
| 11 | 社区文档中文版上线 | 新增贡献指南、贡献者公约、治理机制、安全声明、技术例会及版本发布策略等中文社区文档，并接入文档站导航。 | [贡献指南](./community/CONTRIBUTING_zh.md) |

## 2026年04月

| 序号 | 更新点 | 更新说明 | 相关文档 |
| --- | --- | --- | --- |
| 1 | 更新《安装指南》 | 更新 LLVM 补丁链接；补充 pip 安装覆盖问题的规避说明。 | [安装指南](./installation_guide.md) |
| 2 | 更新《架构设计与核心特性》 | 修订架构设计文档内容，优化章节表述。 | [架构设计与核心特性](./architecture_design_and_core_features.md) |
| 3 | 统一产品命名表述 | 文档中 CANN 类型相关表述统一使用 a2/a3 命名，修正不一致的表述问题。 | / |
| 4 | 更新《算子迁移指南》 | 补充 GPU 到 NPU 的算子迁移说明。 | [算子迁移指南](./migration_guide/index.md) |
| 5 | 更新《性能调优指南》 | 更新 profiling 性能调优相关内容。 | [性能调优指南](./debug_guide/profiling.md) |
| 6 | 文档批量勘误与构建优化 | 修正文档低错与格式问题，优化文档构建流程（doc-tools、docs CI）。 | / |
