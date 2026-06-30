:orphan:

triton.language.extra.cann.extension
====================================

.. currentmodule:: triton.language.extra.cann.extension

Core Types
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    scope
    ascend_address_space
    compile_hint
    multibuffer
    parallel

Data Movement
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    copy_from_ub_to_l1
    fixpipe

Synchronization
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sync_block_all
    sync_block_set
    sync_block_wait
    debug_barrier

Vector Operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sub_vec_id
    sub_vec_num
    conv1d

Enums
-----

.. autosummary::
    :toctree: generated
    :nosignatures:

    PIPE
    MODE
    CORE
    SYNC_IN_VF
    IteratorType
    FixpipeDMAMode
    FixpipeDualDstMode
    FixpipePreQuantMode
    FixpipePreReluMode

Vector/Memory Extension Ops
--------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    extract_slice
    insert_slice
    get_element
    sort
    flip
    cast
    index_put
    gather_out_to_ub
    scatter_ub_to_out
    index_select_simd

Custom Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    custom
    custom_semantic
    register_custom_op

IR Affine Types
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    affine_expr
    affine_constant_expr
    affine_dim_expr
    affine_symbol_expr
    affine_binary_op_expr
    affine_map
