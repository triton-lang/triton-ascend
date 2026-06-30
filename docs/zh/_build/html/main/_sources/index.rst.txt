
欢迎查看 Triton Ascend 文档
============================

**Triton-Ascend** 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，提供高效的核函数自动调优、算子编译及部署能力，支持 Ascend Atlas A2/A3 等系列产品， 兼容 Triton 核心语法的同时，针对昇腾 NPU 特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

.. raw:: html

    <ul>
    <li><a href="https://gitcode.com/Ascend/triton-ascend" target="_blank">GitCode 仓库</a></li>
    <li><a href="https://github.com/triton-lang/triton-ascend" target="_blank">GitHub 仓库</a></li>
    <li><a href="https://triton-ascend.readthedocs.io" target="_blank">Triton Ascend 文档</a></li>
    </ul>

快速开始
----------------------

- :doc:`快速入门 <quick_start>` — 环境要求与环境搭建
- :doc:`安装指南 <installation_guide>` — 安装方式与安装步骤
- :doc:`开发教程 <programming_guide/triton_operator_development_guide>` — Triton算子开发指南

开发指南
---------------------

- :doc:`Triton-Ascend算子开发 <examples/01_vector_add_example>` — 调用新算子开发
- :doc:`Triton-Ascend算子迁移 <migration_guide/index>` — GPU Triton算子迁移
- :doc:`Triton-Ascend算子调试与调优 <debug_guide/index>` — Triton-Ascend autotune 使用指南

更多
------------

- :doc:`贡献指南 <community/CONTRIBUTING_zh>`
- :doc:`常见问题 <FAQ>`


.. toctree 驱动侧栏导航。

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 快速开始
   
   版本说明 <release_note>
   快速入门 <quick_start>
   安装指南 <installation_guide>
   教程 <programming_guide/index>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 特性说明

   架构设计与核心特性 <architecture_design_and_core_features>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 开发指南

   调用新开发算子 <examples/01_vector_add_example>
   Triton-Ascend算子迁移 <migration_guide/index>
   Triton-Ascend算子调试与调优 <debug_guide/index>
   典型算子样例<examples/index>

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
   :caption: 常见问题

   Triton-Ascend FAQ <FAQ>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 贡献指南

   贡献指南 <community/CONTRIBUTING_zh>
   RodMap指导 <community/roadmap_guide>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 社区治理

   贡献者公约 <community/CODE_OF_CONDUCT>
   治理机制 <community/GOVERNANCE_zh>
   技术例会 <community/community_technical_meeting>
   版本发布策略 <community/release_policy>
   Maintainers <community/MAINTAINERS>
   Contributors <community/contributor>
   安全声明 <community/SECURITYNOTE_zh>
