Triton Ascend
============================

.. raw:: html

    <p align="center">
      <a href="https://deepwiki.com/triton-lang/triton-ascend">
        <img src="https://img.shields.io/badge/Ask_AI-DeepWiki-0052D9?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZD0iTTIwIDJINGEyIDIgMCAwIDAtMiAydjEyYTIgMiAwIDAgMCAyIDJoMTRsNCA0VjRhMiAyIDIgMCAwIDAtMi0yeiIgZmlsbD0id2hpdGUiLz48L3N2Zz4%3D" alt="Ask AI on DeepWiki">
      </a>
    </p>

**Triton-Ascend** 是适配华为Ascend处理器的Triton优化版本,主要用于提供高效的核函数自动调优、算子编译及部署能力,支持Ascend Atlas A2/A3/950系列产品,兼容Triton核心语法的同时,针对昇腾NPU特性进行了深度优化,包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

.. raw:: html

    <ul>
    <li><a href="https://gitcode.com/Ascend/triton-ascend" target="_blank">GitCode 仓库</a></li>
    <li><a href="https://github.com/triton-lang/triton-ascend" target="_blank">GitHub 仓库</a></li>
    <li><a href="https://triton-ascend.readthedocs.io" target="_blank">Triton Ascend 文档</a></li>
    </ul>

文档目录
------------

以下是完整的文档目录，包含各章节的详细内容：

**从这里开始**

- :doc:`版本说明 <release_note>`
- :doc:`快速入门 <quick_start>`
- :doc:`安装指南 <installation_guide>`

**教程与样例**

- :doc:`Vector算子开发 <programming_guide/vector_operator>`
- :doc:`Cube算子开发 <programming_guide/cube_operator>`
- :doc:`融合算子开发 <programming_guide/cv_fusion_operator>`
- :doc:`Triton-Ascend autotune <autotune_guide>`
- :doc:`典型算子样例 <examples/index>`
    - :doc:`01_vector_add_example <examples/01_vector_add_example>` — 简单加法示例
    - :doc:`02_fused_softmax_example <examples/02_fused_softmax_example>` — Softmax 融合算子示例
    - :doc:`03_layer_norm_example <examples/03_layer_norm_example>` — Layer Normalization 示例
    - :doc:`04_fused_attention_example <examples/04_fused_attention_example>` — Flash Attention v2 融合注意力算法示例
    - :doc:`05_matrix_multiplication_example <examples/05_matrix_multiplication_example>` — 矩阵乘法高效实现示例
    - :doc:`06_autotune_example <examples/06_autotune_example>` — 使用 Autotune 进行内核自动调优示例
    - :doc:`07_accuracy_comparison_example <examples/07_accuracy_comparison_example>` — 精度比对示例
    - :doc:`08_max_autotune_example <examples/08_max_autotune_example>` — 使用 max_autotune 进行内核自动调优示例
    - :doc:`09_costmodel_example <examples/09_costmodel_example>` — Costmodel 端到端预测示例

**开发指南**

- :doc:`Triton-Ascend算子开发 <programming_guide/index>`
    - :doc:`Vector算子开发 <programming_guide/vector_operator>` — 逐元素、归约、Gather/Scatter 等算子
    - :doc:`Cube算子开发 <programming_guide/cube_operator>` — 矩阵乘、批量矩阵乘等算子
    - :doc:`融合算子开发 <programming_guide/cv_fusion_operator>` — Cube 计算和 Vector 后处理融合场景
- :doc:`Triton-Ascend算子迁移 <migration_guide/index>`
    - :doc:`架构差异分析 <migration_guide/architecture_difference>` — GPU与NPU平台的架构差异
    - :doc:`GPU Triton算子迁移 <migration_guide/migrate_from_gpu>` — 详细的迁移步骤和常见问题
- :doc:`Triton-Ascend算子调试与调优 <debug_guide/index>`
    - :doc:`调试指南 <debug_guide/debugging>` — Triton-Ascend算子调试方法
    - :doc:`性能分析 <debug_guide/profiling>` — 使用性能分析工具定位性能瓶颈
    - :doc:`精度分析 <debug_guide/precision>` — 算子精度问题分析与解决
- :doc:`环境变量与编译选项 <environment_variable_and_compiler_options_reference>`

**API参考**

- :doc:`triton.language API <triton_api/index>`
- :doc:`triton <triton_api/triton/index>`
- :doc:`libdevice开发者手册 <libdevice/libdevice_developer_guide>`

**特性说明**

- :doc:`架构设计与核心特性 <architecture_design_and_core_features>`

**常见问题**

- :doc:`Triton-Ascend FAQ <FAQ>`

**社区**

- :doc:`贡献指南 <community/CONTRIBUTING_zh>`
- :doc:`贡献者公约 <community/CODE_OF_CONDUCT_zh>`
- :doc:`治理机制 <community/GOVERNANCE_zh>`
- :doc:`技术例会 <community/community_technical_meeting>`
- :doc:`Maintainers <community/MAINTAINERS>`
- :doc:`Contributors <community/contributor>`
- :doc:`安全声明 <community/SECURITYNOTE_zh>`


.. toctree 驱动侧栏导航。

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 从这里开始

   版本说明 <release_note>
   快速入门 <quick_start>
   安装指南 <installation_guide>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 教程与样例

   Vector算子开发 <programming_guide/vector_operator>
   Cube算子开发 <programming_guide/cube_operator>
   融合算子开发 <programming_guide/cv_fusion_operator>
   Triton-Ascend autotune <autotune_guide>
   典型算子样例 <examples/index>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 开发指南

   Triton-Ascend算子开发 <programming_guide/index>
   Triton-Ascend算子迁移 <migration_guide/index>
   Triton-Ascend算子调试与调优 <debug_guide/index>
   环境变量与编译选项 <environment_variable_and_compiler_options_reference>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: API参考

   triton.language API <triton_api/index>
   triton <triton_api/triton/index>
   libdevice开发者手册 <libdevice/libdevice_developer_guide>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 特性说明

   架构设计与核心特性 <architecture_design_and_core_features>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 常见问题

   Triton-Ascend FAQ <FAQ>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 社区

   贡献指南 <community/CONTRIBUTING_zh>
   贡献者公约 <community/CODE_OF_CONDUCT_zh>
   治理机制 <community/GOVERNANCE_zh>
   技术例会 <community/community_technical_meeting>
   Maintainers <community/MAINTAINERS>
   Contributors <community/contributor>
   安全声明 <community/SECURITYNOTE_zh>
