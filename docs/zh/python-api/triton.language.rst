:orphan:

triton.language
================

.. currentmodule:: triton.language

Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs
    map_elementwise

Creation Ops
------------

.. autosummary::
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled

Memory/Pointer Ops
------------------

.. autosummary::
    :toctree: generated
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
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d
    gather

Math Ops
--------

.. autosummary::
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
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
    :toctree: generated
    :nosignatures:

    range
    static_range
    condition

Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of

Debug Ops
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert

Inline Assembly
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise
