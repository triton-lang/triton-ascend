# Triton Ascend 最新动态

本文介绍Triton Ascend官方文档的最新上线、新增及变更动态，便于用户及时了解文档更新信息。

软件版本的发布计划、维护策略及配套版本信息不属于本文范围，请参见[版本发布](./release_note.md)。

> **安全声明**：如果您在使用过程中发现安全问题，请按照[安全声明](./community/SECURITYNOTE_zh.md)中提供的渠道进行反馈，请勿在公开渠道披露。

## 2026年10月

| 序号 | 更新点 | 更新说明 | 相关文档 |
| --- | --- | --- | --- |
| 1 | 新增《msDebug 使用指南》《Warmup 使用指南》 | 新增 msDebug 调试工具使用指南与 kernel.warmup 预热编译行为说明，并接入文档站导航索引。 | [msDebug 使用指南](./debug_guide/triton_msdebug_user_guide.md) 、[Warmup 使用指南](./debug_guide/warmup_guide.md) |
| 2 | 更新《Autotune 指南》 | 新增"Triton Ascend 相比社区 autotune 的扩展"（多 config 并行编译、profiler 采集 kernel 性能）与"自动 Tiling 生成能力的范围与行为"等章节。 | [Autotune 指南](./autotune_guide.md) |
| 3 | 更新《安装指南》 | 适配 3.6.0 配套软件包版本（CANN 9.2.0、torch_npu 2.10.0），新增 `--no-build-isolation` 构建提速提示。 | [安装指南](./installation_guide.md) |
| 4 | 更新《libdevice API 文档》 | 统一 libdevice 文档示例的精度断言至 1e-4，并移除示例中的 `TRITON_ENABLE_LIBDEVICE_SIMT` 样板代码，提升示例规范性与一致性。 | [libdevice API](https://triton-ascend.readthedocs.io/zh-cn/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 5 | 更新《环境变量与编译选项》 | 新增 `triton.knobs` 接口章节，并更新新旧编译选项的映射关系与迁移说明。 | [环境变量与编译选项](./environment_variable_and_compiler_options_reference.md) |
