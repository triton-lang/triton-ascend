triton.language.extra.tla
=========================

.. currentmodule:: triton.language.extra.tla

TLA（Triton Language for Ascend）是面向 Ascend NPU 的显式片上内存、SIMD
向量寄存器和流水/核间同步编程接口。

本页由 Sphinx autodoc 根据各模块的 ``__all__`` 和源码 docstring 自动生成。
新增接口时只需将其加入对应模块的 ``__all__`` 并补充 docstring，无需手工修改
本页。平台约束和示例由 ``_ascend_constraints.py`` 注入，示例源码位于
``docs/zh/examples/tla``。

当前页面覆盖 ``tla`` 顶层接口，以及 ``tla.scope``、``tla.simd``、``tla.cube``、
``tla.dma``、``tla.sync`` 和 ``tla.math`` 等子模块的对外接口。

编程模型和完整内核示例请参考 :doc:`../triton_api_extension/tla`。

顶层接口
--------

.. automodule:: triton.language.extra.tla
    :members:
    :undoc-members:
    :exclude-members: simd, cube, sync, math, scope, to_ir, mangle, shape, dtype
    :member-order: bysource

核心作用域
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    scope
    scope.vector
    scope.simd
    scope.cube

SIMD 向量寄存器
---------------

.. automodule:: triton.language.extra.tla.simd
    :members:
    :undoc-members:
    :member-order: bysource

Cube 接口
---------

.. automodule:: triton.language.extra.tla.cube
    :members:
    :undoc-members:
    :member-order: bysource

DMA 接口
--------

.. automodule:: triton.language.extra.tla.dma

``tla.dma`` 提供显式的 GM / UB / L1 / L0 DMA 路由，例如
``load_gm_to_ub``、``load_gm_to_l1``、``copy_ub_to_l1``、``copy_l1_to_l0``、
``store_l0c_to_gm`` 和 ``set_ctrl``。这些名称同时从顶层
``triton.language.extra.tla`` 导出，签名与约束见上方“顶层接口”。

同步接口
--------

.. automodule:: triton.language.extra.tla.sync
    :members:
    :undoc-members:
    :member-order: bysource

数学接口
--------

.. automodule:: triton.language.extra.tla.math

``tla.math`` 提供 ``sqrt``、``truncmod`` 和 ``compensated_rstd``。
这些名称同时从顶层 ``triton.language.extra.tla`` 导出。
