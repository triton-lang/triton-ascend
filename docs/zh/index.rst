Triton Ascend
============================

**Triton-Ascend** 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，提供高效的核函数自动调优、算子编译及部署能力，支持 Ascend Atlas A2/A3 等系列产品， 兼容 Triton 核心语法的同时，针对昇腾 NPU 特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。


文档目录
--------

快速开始
~~~~~~~~

- :doc:`快速入门 <quick_start>` — 环境要求与环境搭建
- :doc:`安装指南 <installation_guide>` — 安装方式与安装步骤

特性说明
~~~~~~~~

- :doc:`架构设计与核心特性 <architecture_design_and_core_features>`

Triton-Ascend算子开发指南
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`Triton-Ascend算子开发 <programming_guide>`

Triton-Ascend算子迁移指南
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`昇腾与GPU的开发差异 <migration_guide/architecture_difference>`
- :doc:`GPU Triton算子迁移 <migration_guide/migrate_from_gpu>`

典型算子样例
~~~~~~~~~~~~

- :doc:`Tutorials <examples/index>`

Triton-Ascend算子调试与调优
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`Triton-Ascend调试指南 <debug_guide/debugging>`
- :doc:`Triton-Ascend性能分析方法 <debug_guide/profiling>`

Triton API 接口说明
~~~~~~~~~~~~~~~~~~~

- :doc:`triton <python-api/triton>`
- :doc:`triton.language <python-api/triton.language>`
- :doc:`triton.testing <python-api/triton.testing>`

昇腾扩展 API 接口说明
~~~~~~~~~~~~~~~~~~~~~

- :doc:`triton.language.extra.cann.extension <python-api/triton.language.extra.cann.extension>`
- :doc:`triton.language.extra.cann.libdevice <python-api/triton.language.extra.cann.libdevice>`
- :doc:`triton.extension.buffer.language <python-api/triton.extension.buffer.language>`

环境变量
~~~~~~~~

- :doc:`环境变量 <environment_variable_reference>`

常见问题
~~~~~~~~

- :doc:`Triton-Ascend FAQ <FAQ>`

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 快速开始

   快速入门 <quick_start>
   安装指南 <installation_guide>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 特性说明

   架构设计与核心特性 <architecture_design_and_core_features>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Triton-Ascend算子开发指南

   Triton-Ascend算子开发 <programming_guide>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Triton-Ascend算子迁移指南

   昇腾与GPU的开发差异 <migration_guide/architecture_difference>
   GPU Triton算子迁移 <migration_guide/migrate_from_gpu>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 典型算子样例

   Tutorials <examples/index>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Triton-Ascend算子调试与调优

   Triton-Ascend调试指南 <debug_guide/debugging>
   Triton-Ascend性能分析方法 <debug_guide/profiling>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Triton API 接口说明

   triton <python-api/triton>
   triton.language <python-api/triton.language>
   triton.testing <python-api/triton.testing>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 昇腾扩展 API 接口说明

   triton.language.extra.cann.extension <python-api/triton.language.extra.cann.extension>
   triton.language.extra.cann.libdevice <python-api/triton.language.extra.cann.libdevice>
   triton.extension.buffer.language <python-api/triton.extension.buffer.language>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 环境变量

   环境变量 <environment_variable_reference>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 常见问题

   Triton-Ascend FAQ <FAQ>
