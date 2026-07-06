# Triton-Ascend max_autotune Usage Guide

## Document Purpose

This document is intended for users already familiar with Triton-Ascend's autotuning mechanism, introducing advanced usage of the `max_autotune` decorator:

- Differences and relationships between `max_autotune` and standard `@triton.autotune`;
- How to leverage the Cartesian product expansion mechanism to batch generate candidate configurations;
- Tuning parameters supported by different `kernel_type` values and their applicable scenarios;
- When to choose `max_autotune` over manually written configuration lists.

## Quick Start

`max_autotune` is an extended version of the `@triton.autotune` decorator. It allows Cartesian product expansion of each base configuration with additional tuning parameters before autotuning, significantly reducing the number of configurations users need to manually enumerate.

### 1. Basic Usage

```python
import triton
import triton.language as tl
from triton.backends.ascend.runtime import max_autotune


@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 256}),
    ],
    key=["M", "N"],
    kernel_type="mix",
    # Additional tuning parameters, each value must be a list
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4],
)
@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    **META
):
    ...
```

The equivalent number of configurations after expansion:

- Base configurations: 2
- User-provided tuning parameters: `enable_hivm_auto_cv_balance` (2 values), `tile_mix_vector_loop` (2 values)
- Parameters using default values: `set_workspace_multibuffer` (2 values), `tile_mix_cube_loop` (2 values), other parameters have 1 value each

Total configuration count: `2 × 2 × 2 × 2 × 2 = 32` configurations.

> Note: `kernel_type="mix"` supports many parameters. Parameters not explicitly provided will use default values for expansion. If you want a specific parameter to not participate in expansion, you can fix its value in the base `Config`'s `kwargs`.

### 2. Relationship between `max_autotune` and `@triton.autotune`

`max_autotune` essentially calls `get_max_configs` to expand base configurations first, then passes the expanded configuration list to the standard `@triton.autotune`. Therefore:

- All parameters supported by `@triton.autotune` (`key`, `prune_configs_by`, `reset_to_zero`, etc.) are also valid in `max_autotune`;
- `max_autotune` additionally provides the `kernel_type` parameter and `**tuning_params` tuning parameter space;
- The benchmarking, selection, and caching are still performed by `@triton.autotune`.

### 3. Must Import Ascend Backend Extension

Like using `configs=[]` for automatic Tiling, `max_autotune` requires Ascend backend extension support. `max_autotune` must be imported separately from the Ascend backend module:

```python
from triton.backends.ascend.runtime import max_autotune
```

## Kernel Types and Supported Parameters

`max_autotune` distinguishes different operator types through the `kernel_type` parameter, with each type supporting a different set of tuning parameters.

### Parameter Support Matrix

| Parameter | cube | mix | vector | Default Value | Valid Values | Description |
|------|:----:|:---:|:------:|--------|--------|------|
| `num_stages` | ✅ | ✅ | ✅ | `[2]` | `[1, 2]` | Number of pipeline stages |
| `unit_flag` | ✅ | ✅ | ❌ | `[False]` | Boolean list | Cube搬出 related synchronization optimization |
| `limit_auto_multi_buffer_of_local_buffer` | ✅ | ✅ | ❌ | `["no-l0c"]` | `["no-limit", "no-l0c"]` | Configure the scope of automatic multi-buffer for local buffer |
| `limit_auto_multi_buffer_only_for_local_buffer` | ❌ | ✅ | ❌ | `[False]` | Boolean list | Limit automatic multi-buffer to only apply to local buffer |
| `set_workspace_multibuffer` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4]` | Configure workspace multi-buffer levels |
| `enable_hivm_auto_cv_balance` | ❌ | ✅ | ❌ | `[True]` | Boolean list | Enable or disable automatic CV balance |
| `tile_mix_vector_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | Configure the number of splits for Vector loop |
| `tile_mix_cube_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | Configure the number of splits for Cube loop |
| `enable_ubuf_saving` | ❌ | ✅ | ✅ | `[True]` | Boolean list | Whether to enable ubuf saving |

### Kernel Type Description

- **cube**: Pure cube (matrix multiplication-like) operators, supporting the fewest tuning parameters;
- **vector**: Pure vector operators, only supporting `num_stages` and `enable_ubuf_saving`;
- **mix**: Mixed cube+vector operators (default type), supporting the most complete set of tuning parameters.

## Parameter Value Priority and Expansion Logic

### Parameter Value Priority

The values of tuning parameters are determined by the following priority:

1. **tuning_params parameter** (highest priority): Candidate value lists passed through `**tuning_params`;
2. **Value in base configuration**: If the parameter already exists in the base `Config`'s `kwargs`, it is fixed to that value (converted to a single-element list);
3. **Default value** (lowest priority): Obtained from the internal default value table.

### Expansion Example

Assume the following configuration:

```python
@max_autotune(
    configs=[triton.Config(kwargs={'BLOCK_M': 128}, num_stages=2)],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
)
@triton.jit
def kernel(...):
    ...
```

Expansion process:

1. `kernel_type="vector"` supports parameters `num_stages` and `enable_ubuf_saving`;
2. `num_stages` is provided in `tuning_params` as `[1, 2]`, with highest priority;
3. `enable_ubuf_saving` is not provided, using default value `[True]`;
4. Cartesian product expansion yields 2 configurations.

Expansion result is equivalent to:

```python
configs=[
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=1),
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=2),
]
```

### Fixing Parameters in Base Configuration

If you want a specific tuning parameter to not participate in expansion, you can fix it directly in the base configuration:

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': False}),
    ],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
    # enable_ubuf_saving is fixed to False in base configuration, will not use default value [True]
)
@triton.jit
def kernel(...):
    ...
```

## Usage Considerations

### 1. Unsupported Parameters Will Be Ignored

If you pass a parameter through `tuning_params` that is not supported by the current `kernel_type`, a warning will be generated and the parameter will be ignored:

```python
# Warning: tile_mix_vector_loop is not supported for kernel_type="vector"
@max_autotune(
    configs=[...],
    key=["M"],
    kernel_type="vector",
    tile_mix_vector_loop=[2, 4],  # Will be ignored with a warning
)
@triton.jit
def kernel(...):
    ...
```

### 2. Parameter Values Must Be Lists

Each value in `tuning_params` must be a list or tuple, and cannot be empty:

```python
# Correct format
enable_hivm_auto_cv_balance=[True, False]

# Incorrect format: not a list
enable_hivm_auto_cv_balance=True  # Will cause validation error
```

### 3. Configuration Count Growth

The number of expanded configurations equals:
Base configuration count × Π(length of each tuning_param list)

For example, 2 base configurations × 3 parameters (list lengths of 2, 3, 2) = 12 expanded configurations.

Too many configurations will increase the initial tuning time. It is recommended to reasonably control the parameter space based on actual needs.

### 4. Difference from `configs=[]`

`max_autotune` and `@triton.autotune(configs=[], ...)` are two different autotuning strategies:

| Feature | `max_autotune` | `@triton.autotune(configs=[])` |
|------|----------------|--------------------------------|
| Tiling parameter generation | Requires user to specify in base configurations | Automatically generated by Ascend backend |
| Compilation parameter tuning | Supports expansion through `tuning_params` | Passed through `hints` parameter |
| Applicable scenarios | Clearly know Tiling parameter space, need to tune compilation parameters | Want Tiling parameters to also be automatically generated |

## Advanced Usage

### 1. Combining Multiple Tuning Parameters

For mixed type operators (`kernel_type="mix"`), you can simultaneously tune multiple parameters:

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}),
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
    ],
    key=["M", "N", "K"],
    kernel_type="mix",
    num_stages=[1, 2],
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4, 8],
    tile_mix_cube_loop=[2, 4],
)
@triton.jit
def mixed_kernel(...):
    ...
```

Expanded configuration count calculation:

- Base configurations: 2
- User-provided parameters: `num_stages` (2), `enable_hivm_auto_cv_balance` (2), `tile_mix_vector_loop` (3), `tile_mix_cube_loop` (2)
- Parameters using default values: `set_workspace_multibuffer` (default `[2, 4]` → 2), other parameters have 1 value each

Total configuration count: `2 × 2 × 2 × 2 × 3 × 2 = 96` configurations.

### 2. Tuning for Cube Type Operators

Cube type operators (such as pure matrix multiplication) support fewer parameters:

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=["M", "N", "K"],
    kernel_type="cube",
    num_stages=[1, 2],
    unit_flag=[True, False],
    limit_auto_multi_buffer_of_local_buffer=["no-limit", "no-l0c"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

### 3. Tuning for Vector Operators

Vector operators support the fewest tuning parameters:

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}),
    ],
    key=["N"],
    kernel_type="vector",
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def vector_kernel(...):
    ...
```

## Summary

`max_autotune` is an advanced autotuning tool provided by Triton-Ascend, suitable for the following scenarios:

1. Known Tiling parameter space, want to reduce the workload of manually enumerating configurations;
2. Need to jointly tune multiple Ascend compilation parameters (such as `num_stages`, `enable_hivm_auto_cv_balance`, etc.);
3. Want to batch generate candidate configurations through Cartesian product.

The core value of `max_autotune` lies in: using a small number of base configurations + tuning parameter space description to automatically expand into a complete set of candidate configurations, balancing flexibility and convenience.
