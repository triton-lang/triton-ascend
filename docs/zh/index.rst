Triton Ascend
============================

.. raw:: html

    <p align="center">
      <a href="https://deepwiki.com/triton-lang/triton-ascend">
        <img src="https://deepwiki.com/badge.svg" alt="Ask AI on DeepWiki">
      </a>
    </p>

**Triton-Ascend** 是适配华为Ascend处理器的Triton优化版本,主要用于提供高效的核函数自动调优、算子编译及部署能力,支持Ascend Atlas A2/A3/950系列产品,兼容Triton核心语法的同时,针对昇腾NPU特性进行了深度优化,包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

文档目录
------------

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

**开发指南**

- :doc:`Triton-Ascend算子开发 <programming_guide/index>`
- :doc:`Triton-Ascend算子迁移 <migration_guide/index>`
- :doc:`Triton-Ascend算子调试与调优 <debug_guide/index>`
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
- :doc:`Contributors <community/CONTRIBUTOR>`
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
   Contributors <community/CONTRIBUTOR>
   安全声明 <community/SECURITYNOTE_zh>
