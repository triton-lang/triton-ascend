:orphan:

triton.language
================

.. currentmodule:: triton.language

Programming Model
-----------------

.. autosummary::
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs
    map_elementwise

Creation Ops
------------

.. autosummary::
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast

Shape Manipulation Ops
----------------------

.. autosummary::
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view

Linear Algebra Ops
------------------

.. autosummary::
    :nosignatures:

    dot
    dot_scaled

Memory/Pointer Ops
------------------

.. autosummary::
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance

Indexing Ops
------------

.. autosummary::
    :nosignatures:

    flip
    where
    swizzle2d
    gather

Math Ops
--------

.. autosummary::
    :nosignatures:

    abs
    add
    cdiv
    ceil
    clamp
    cos
    div
    div_rn
    erf
    exp
    exp2
    fdiv
    floordiv
    floor
    fma
    log
    log2
    maximum
    minimum
    mod
    mul
    neg
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    sub
    umulhi

Logical Ops
-----------

.. autosummary::
    :nosignatures:

    and
    or
    xor
    not
    logical_and
    logical_or
    invert
    lshift
    rshift

Comparison Ops
--------------

.. autosummary::
    :nosignatures:

    gt
    ge
    lt
    le
    eq
    ne

Reduction Ops
-------------

.. autosummary::
    :nosignatures:

    argmax
    argmin
    max
    min
    reduce
    reduce_or
    sum
    xor_sum

Scan/Sort Ops
-------------

.. autosummary::
    :nosignatures:

    associative_scan
    bitonic_merge
    cumprod
    cumsum
    histogram
    sort
    topk

Atomic Ops
----------

.. autosummary::
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_max
    atomic_min
    atomic_or
    atomic_xchg
    atomic_xor

Random Number Generation
------------------------

.. autosummary::
    :nosignatures:

    randint4x
    randint
    rand
    rand4x
    randn
    randn4x

Iterators
---------

.. autosummary::
    :nosignatures:

    range
    static_range
    condition

Compiler Hint Ops
-----------------

.. autosummary::
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of

Debug Ops
---------

.. autosummary::
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert

Inline Assembly
---------------

.. autosummary::
    :nosignatures:

    inline_asm_elementwise


.. toctree::
    :maxdepth: 1
    :class: sidebar-groups-only

    triton.language/programming_model
    triton.language/creation_ops
    triton.language/shape_manipulation_ops
    triton.language/linear_algebra_ops
    triton.language/memory_pointer_ops
    triton.language/indexing_ops
    triton.language/math_ops
    triton.language/logical_ops
    triton.language/comparison_ops
    triton.language/reduction_ops
    triton.language/scan_sort_ops
    triton.language/atomic_ops
    triton.language/random_number_generation
    triton.language/iterators
    triton.language/compiler_hint_ops
    triton.language/debug_ops
    triton.language/inline_assembly
