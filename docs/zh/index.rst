Triton Ascend
=============

.. raw:: html

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend" data-show-count="true" data-size="large" aria-label="Star triton-lang/triton-ascend on GitHub">Star</a>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch triton-lang/triton-ascend on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/triton-lang/triton-ascend/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork triton-lang/triton-ascend on GitHub">Fork</a>
   </p>

   <p style="text-align:center; margin-top: 4px;">
   <a href="https://deepwiki.com/triton-lang/triton-ascend"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
   </p>

**Triton-Ascend** 是适配华为Ascend处理器的Triton优化版本，主要用于提供高效的核函数自动调优、算子编译及部署能力，支持Ascend Atlas A2/A3/950系列产品，兼容Triton核心语法的同时，针对昇腾NPU特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 从这里开始

   版本说明 <release_note>
   快速入门 <quick_start>
   安装指南 <installation_guide>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 教程与样例

   Vector算子开发 <programming_guide/vector_operator>
   Cube算子开发 <programming_guide/cube_operator>
   融合算子开发 <programming_guide/cv_fusion_operator>
   Triton-Ascend autotune <autotune_guide>
   典型算子样例 <examples/index>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 开发指南

   Triton-Ascend算子开发 <programming_guide/index>
   Triton-Ascend算子迁移 <migration_guide/index>
   Triton-Ascend算子调试与调优 <debug_guide/index>
   环境变量与编译选项 <environment_variable_and_compiler_options_reference>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Triton API 接口说明

   triton <triton_api/triton/index>
   triton.language <triton_api/index>
   triton.testing <python-api/triton.testing>
   算子责任人清单 <triton_api/op_owner_list>
   libdevice开发者手册 <libdevice/libdevice_developer_guide>

.. .. toctree::
..    :maxdepth: 1
..    :titlesonly:
..    :caption: 昇腾扩展 API 接口说明

..    triton.language.extra.cann.extension <python-api/triton.language.extra.cann.extension>
..    triton.language.extra.cann.libdevice <python-api/triton.language.extra.cann.libdevice>
..    triton.extension.buffer.language <python-api/triton.language.extra.extension.buffer.language>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 特性说明

   架构设计与核心特性 <architecture_design_and_core_features>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 常见问题

   Triton-Ascend FAQ <FAQ>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: 社区

   贡献指南 <community/CONTRIBUTING_zh>
   贡献者公约 <community/CODE_OF_CONDUCT_zh>
   治理机制 <community/GOVERNANCE_zh>
   技术例会 <community/community_technical_meeting>
   Maintainers <community/MAINTAINERS>
   Contributors <community/CONTRIBUTOR>
   安全声明 <community/SECURITYNOTE_zh>
