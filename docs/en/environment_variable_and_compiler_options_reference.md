# Environment Variables and Compiler Options

This document summarizes Triton-Ascend behavior controls that developers can set explicitly, including environment variables configured before running a program and NPU compiler options passed through `triton.Config` or kernel launch meta-parameters.

## Environment Variables

### Environment Variable Usage Example

Set environment variables before starting the Python program. Example:

```bash
export TRITON_DEBUG=1
python run_kernel.py
```

### Environment Variable Reference Table

The following table describes how to set environment variables.

| Category| Environment Variable| Default Value| Function Description| Setting Description| Change Description|
|------|----------|------|----------|----------|----------|
| **Debugging and logging**| TRITON_DEBUG | **0** or not set| Specifies whether to enable the debugging output function of Triton to print detailed debugging information during running. This is useful for troubleshooting problems in the compilation or execution phase. When this parameter is set to **1**, Triton outputs more information about the compilation, kernel generation, and execution. Some implementations may support more fine-grained debugging levels (such as 2 and 3), depending on the Triton version and implementation.| **0**: The debugging is disabled.<br>**1**: The debugging is enabled.| |
| **Debugging and logging**| MLIR_ENABLE_DUMP | **0** or not set| Specifies whether to dump the intermediate representation (IR) of all kernels before each MLIR optimization. You can set `MLIR_ENABLE_DUMP` to `kernelName` to dump the IR of a specific kernel.| **0**: Do not dump.<br>**1**: Dump the IR of all kernels.<br>*kernelName*: Dump the IR of a specific kernel.| The Triton cache may interfere with the dump. If `MLIR_ENABLE_DUMP=1` does not take effect, you can run `rm -r ~/.triton/cache` to clear the Triton cache.|
| **Debugging and logging**| LLVM_IR_ENABLE_DUMP | **0** or not set| Specifies whether to dump the IR before each LLVM IR optimization.| **0**: Do not dump.<br>**1**: Dump IRs.| |
| **Debugging and logging**| TRITON_REPRODUCER_PATH | Not set| Generates the MLIR reproduction file before each MLIR compilation phase. If a phase fails, `<reproducer_path>` saves the MLIR status before the failure.| `<reproducer_path>`: save path.| |
| **Debugging and logging**| TRITON_INTERPRET | **0** or not set| Specifies whether to use the Triton interpreter instead of the GPU for running and support inserting Python breakpoints in kernel function code.| **0**: Breakpoints are not supported.<br>**1**: Breakpoints are supported.| |
| **Debugging and logging**| TRITON_ENABLE_LLVM_DEBUG | **0** or not set| Specifies whether to pass the`-debug` parameter to LLVM and outputs a large amount of debugging information. If there is too much information, you can use `TRITON_LLVM_DEBUG_ONLY` to limit the output scope.| **0**: Do not pass.<br>**1**: Pass.| Another method to reduce output interference is as follows: Set the running program by setting `LLVM_IR_ENABLE_DUMP` to `1`, extract the IR before the target LLVM optimization channel, and run the `opt` tool of the LLVM separately. In this case, you can add `-debug-only=foo` to the command line to limit the debugging range.|
| **Debugging and logging**| TRITON_LLVM_DEBUG_ONLY | Not set| Equivalent to the `-debug-only` command line option of LLVM. This parameter can be used to limit the LLVM debugging output to a specific optimization channel or component name (defined by the `#define DEBUG_TYPE` macro in LLVM and Triton), thereby effectively reducing redundant debugging output. You can specify one or more comma-separated values, for example, `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`.| Comma-separated values: channel or component name| |
| **Debugging and logging**| USE_IR_LOC | **0** or not set| Specifies whether to include location information (such as file names and line numbers) in the generated IR. This information is helpful for debugging, but may increase the size of the generated IR. If this parameter is set to **1**, the IR is re-parsed, and the location information is mapped to the line number of the IR file with a specific extension (not the line number of the Python source file). This enables a direct mapping from the IR to the LLVM IR/PTX. When used with the performance analysis tool, this parameter can be used to implement fine-grained performance analysis on IR instructions.| **0**: No location information is included.<br>**1**: The location information is included.| |
| **Debugging and logging**| TRITON_DISABLE_LINE_INFO | **true**| Specifies whether to append `--enable-debug-info=true` to the `bishengir-compile` command, that is, whether debugging information such as line numbers is included in the generated kernel executable file. **Note**: The default value of Triton-Ascend is `true` (line number information is **disabled** by default, which is the opposite of community Triton, where it is enabled by default). Line number information can be used to locate accuracy/performance issues and for profiling analysis, but it increases the size of the compiled artifact.| **0/false**: Line number information is enabled.<br>**1/true**: Line number information is disabled.| When used with performance analysis, see [Operator Performance Tuning Method](./debug_guide/profiling.md).|
| **Debugging and logging**| TRITON_PRINT_AUTOTUNING | **0** or not set| After the automatic optimization is complete, the optimal configuration and total time of each kernel are output.| **0**: Do not output.<br>**1**: Output.| |
| **Debugging and logging**| MLIR_ENABLE_REMARK | **0** or not set| Specifies whether to enable the output of remarks during MLIR compilation, including performance warnings in remarks.| **0**: Disabled.<br>**1**: Enabled.| |
| **Debugging and logging**| TRITON_KERNEL_DUMP | **0** or not set| Specifies whether to enable the dump function of the Triton kernel. When this function is enabled, Triton saves the generated kernel code (IR and final PTX in each compilation phase) to the specified directory.| **0**: Disabled.<br>**1**: Enabled.| |
| **Debugging and logging**| TRITON_DUMP_DIR | ~/.triton/dump | Specifies the directory for storing the Triton kernel dump file, which is the directory for saving the IR and PTX when `TRITON_KERNEL_DUMP` is set to `1`.| **"path"**: save path.| |
| **Debugging and logging**| TRITON_DEVICE_PRINT | **0** or not set| If this parameter is set to `1` or `true` (`TRUE` is converted to `true`), the function of `tl.device_print` is enabled. Note: This function uses the GM buffer (the pointer of which is passed to the kernel).| **0**: Disabled.<br>**1**: The functionality of `tl.device_print` is enabled.| The maximum size of the GM buffer for each thread is 16 KB. If the buffer size exceeds 16 KB, the excess content will be discarded. The value is fixed currently and will be adjusted through an environment variable.|
| **Debugging and logging**| TRITON_MEMORY_DISPLAY | **0** or not set| Specifies whether to generate a JSON file for memory usage. When `TRITON_MEMORY_DISPLAY` is set to `1`, the memory_info_aic/aiv.json files are saved to the current directory.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_ALWAYS_COMPILE | **0** or not set| Specifies whether Triton forcibly recompiles the kernel each time it runs, instead of using the existing cached version. By default, Triton caches the compiled kernels (based on parameters and configurations) to improve performance. If this parameter is set to **1**, Triton ignores the cache and recompiles the kernel each time it runs, which is useful for debugging or testing new compiler features.| **0**: Disabled.<br>**1**: All kernels are recompiled during each running.| |
| **Compilation control**| DISABLE_LLVM_OPT | **0** or not set| If this parameter is set to **1**, the optimization steps (LLVM optimization of **make_llir** and **make_ptx**) during LLVM compilation can be disabled. If this parameter is set to a character string, the LLVM optimization flags to be disabled are parsed. For example, if `DISABLE_LLVM_OPT` is set to `"disable-lsr"`, the loop strength optimization is disabled (this optimization may cause a performance fluctuation of up to 10% in some kernels with register pressure).| **0**: The LLVM optimization is enabled.<br>**1**: The optimization steps (LLVM optimization of make_llir and make_ptx) during LLVM compilation are disabled. <list>:"disable-lsr": Disables loop strength optimization </list>| |
| **Compilation control**| MLIR_ENABLE_TIMING | **0** or not set| Specifies whether to enable the time statistics function during MLIR compilation.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| LLVM_ENABLE_TIMING | **0** or not set| Specifies whether to enable the time statistics function during LLVM compilation.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_DEFAULT_FP_FUSION | **1** (enabled)| Specifies whether to enable the floating-point operation fusion optimization by default. The default floating-point operation fusion behavior (for example, **mul+add->fma**) is overwritten.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_KERNEL_OVERRIDE | **0** or not set| Specifies whether to enable the Triton kernel override function. You can use the user-specified external file (such as IR/PTX) to override the default generated kernel code at the beginning of each compilation phase.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_OVERRIDE_DIR | ~/.triton/override | Specifies the directory for searching the Triton kernel override file. Directory for loading the IR/PTX file when `TRITON_KERNEL_OVERRIDE` is set to `1`.| **"path"**: save path.| |
| **Compilation control**| TRITON_COMPILE_ONLY | **0** or not set| Specifies whether to perform only compilation without execution. This parameter is used when **remote_launch** is used.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_DISABLE_PRECOMPILE | **0** or not set| Specifies whether to disable precompilation.| **0**: Precompilation is enabled.<br>**1**: Precompilation is disabled.| |
| **Running and scheduling**| TRITON_ENABLE_TASKQUEUE | **1**| Specifies whether to enable **task_queue**.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling**| TRITON_ENABLE_SANITIZER | **0** or not set| Specifies whether to enable SANITIZER.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling**| ENABLE_PRINT_UB_BITS | **0** or not set| After this parameter is enabled, the current UB usage can be obtained for the inductor.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling** | NPU_DEVICE_LIMIT | Not set | Users can set the maximum number of ai cores and vector cores for operator runtime. Format: numbers separated by commas.Example:14,28.| No default value | |
| **Others**| TRITON_BENCH_METHOD | Not set| When the Ascend NPU is used, change `do_bench` in `testing.py` to `do_bench_npu`. (This parameter is used when `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE` is set to `1`.) If this parameter is set to `default`, the original `do_bench` function is still called even if the NPU is available.| **"npu"**: Switch to `do_bench_npu`.| |
| **Others**| TRITON_REMOTE_RUN_CONFIG_PATH | path | Specifies the configuration path for remote running.| Specify the path directly.| |

## Compiler Options

Compiler options control the compilation strategy for a single Triton kernel and can be passed through `triton.Config`, autotune parameters, or kernel launch meta-parameters.

### Compiler Option Usage Example

For example, pass `multibuffer` directly during kernel launch:

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024, multibuffer=True)
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((4096,), device="npu", dtype=torch.float32)
    y = torch.randn((4096,), device="npu", dtype=torch.float32)
    out = add(x, y)
    torch.npu.synchronize()
    print(out[:4])
```

### Compiler Option Reference Table

The following table describes the options.

| Category | Compiler Option | Default/Values | Function Description | Setting Description |
|----------|-----------------|----------------|----------------------|--------------------|
| **General pipeline** | `multibuffer` | `True` (default), `False` | Enables or disables ping-pong/double-buffer pipelines. Enabled by default. | `triton.Config` or launch meta-parameter |
| **Graph optimization** | `enable_graph_optimize` | `True` (default), `False` | Enables or disables TTIR Graph Optimization. The backend owns the individual rules, rewrite limit, and UB budget. | `triton.Config` or launch meta-parameter |
| **BiSheng compiler** | `bisheng_options` | Backend default string or a user-provided string | Forwards additional arguments to BiSheng compilation paths that support this option. | `triton.Config` or launch meta-parameter |
| **CV fusion** | `enable_auto_bind_sub_block` | `None`, `True`, `False` | Enables or disables automatic sub-block binding. | `triton.Config` or launch meta-parameter |
| **CV fusion** | `enable_hivm_auto_cv_balance` | `None`, `True`, `False` | Enables or disables automatic CV balance. | `triton.Config` or autotune parameter |
| **CV fusion** | `enable_cube_block_merge` | `False` (default), `True` | Controls Cube block merging in the DynamicCV pipeline. | `triton.Config` or launch meta-parameter |
| **VF fusion** | `vf_fusion_mode` | `None` (default; uses the BiShengIR default of `"max-parallel"`), `"max-parallel"`, `"all-op"`, `"ub-aware-op"` | Selects the VF fusion strategy during BiShengIR compilation on Ascend 950. | `triton.Config` or launch meta-parameter |
| **VF fusion** | `enable_vf_fusion` | `None` (default; uses the BiShengIR default of `True`), `True`, `False` | Controls whether VF fusion is enabled during BiShengIR compilation on Ascend 950. This option is the VF fusion master switch; `vf_fusion_mode` only selects the fusion strategy. | `triton.Config` or launch meta-parameter |
| **HFusion** | `hfusion_enable_multiple_consumer_fusion` | `False` (default), `True` | Controls multiple-consumer fusion during BiShengIR compilation on Ascend 950. | `triton.Config` or launch meta-parameter |
| **CV fusion/sync** | `sync_solver` | `None`, `True`, `False` | Enables or disables the HIVM synchronization solver. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `unit_flag` | `None`, `True`, `False` | Cube-output synchronization option. | `triton.Config` or autotune parameter |
| **Synchronization** | `inject_barrier_all` | `None`, `True`, `False` | Enables or disables automatic barrier synchronization injection. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `inject_block_all` | `None`, `True`, `False` | Enables or disables automatic block synchronization injection. | `triton.Config` or launch meta-parameter |
| **Multibuffer scope** | `limit_auto_multi_buffer_only_for_local_buffer` | `None`, `True`, `False` | Restricts automatic multi-buffering to local buffers. | `triton.Config` or autotune parameter |
| **Multibuffer scope** | `limit_auto_multi_buffer_of_local_buffer` | `None`, `"no-limit"`, `"no-l0c"` | Configures the local-buffer automatic multi-buffering scope. | `triton.Config` or autotune parameter |
| **Workspace** | `set_workspace_multibuffer` | `None`, `2`, `4` | Configures workspace multi-buffering. | `triton.Config` or autotune parameter |
| **CV fusion tiling** | `tile_mix_vector_loop` | `None`, `2`, `4`, `8` | Configures the Vector loop split count. | `triton.Config` or autotune parameter |
| **CV fusion tiling** | `tile_mix_cube_loop` | `None`, `2`, `4`, `8` | Configures the Cube loop split count. | `triton.Config` or autotune parameter |
| **DynamicCV buffering** | `buf_slot_num_of_veccore` | `None` (default) or an integer | Configures the number of vector-core-local buffer slots. | `triton.Config` or launch meta-parameter |
| **DynamicCV buffering** | `buf_slot_num_of_crosscore` | `None` (default) or an integer | Configures the number of cross-core buffer slots. | `triton.Config` or launch meta-parameter |
| **DynamicCV buffering** | `buf_slot_num_of_gm` | `None` (default) or an integer | Configures the number of GM load buffer slots. | `triton.Config` or launch meta-parameter |
| **Compilation mode** | `compile_mode` | `"simd_simt_template"` (default), `"simd"`, `"simt_only"` | Controls SIMD / SIMT compilation. `"simd"`: pure SIMD; `"simd_simt_template"`: the standard SIMD pipeline with template-SIMT subpaths enabled on Ascend 950; `"simt_only"`: the pure-SIMT path (`ttir→npubin`), supported only on Ascend 950. | `triton.Config` or launch meta-parameter |

(compiler-option-cleanup-and-compatibility)=

### Compiler Option Cleanup and Compatibility

The environment-variable and compiler-option reference tables list only active public controls. Environment variables and compiler options that have been removed or moved under backend ownership are no longer listed. As a temporary compatibility measure, using one of those deprecated controls currently emits a `FutureWarning`; its value is ignored and the backend-managed or default behavior is used. These deprecated controls are scheduled for complete removal in a future release. After a deprecated compiler option is removed, continuing to pass it will be treated as an unsupported option and cause compilation to fail. Users should migrate away from deprecated controls during the compatibility period.

#### Renamed Compiler Options

The following deprecated names remain accepted during the compatibility period and are routed to their canonical replacements. New code should use the canonical forms directly:

| Deprecated name | Canonical name or value | Compatibility behavior |
|-----------------|-------------------------|------------------------|
| `force_simt_only` | `compile_mode="simt_only"` | Any explicit use emits a `FutureWarning`; `True` routes and overrides an existing `compile_mode`, while `False` leaves the current mode unchanged. |
| `force_simt_template` | `compile_mode="simd_simt_template"` | Any explicit use emits a `FutureWarning`; `True` routes and overrides an existing `compile_mode`, while `False` leaves the current mode unchanged. |
| `intra_cache_num` | `buf_slot_num_of_veccore` | Emits a `FutureWarning` and preserves the value; the canonical name wins when both names are supplied. |
| `inter_cache_num` | `buf_slot_num_of_crosscore` | Emits a `FutureWarning` and preserves the value; the canonical name wins when both names are supplied. |
| `load_cache_num` | `buf_slot_num_of_gm` | Emits a `FutureWarning` and preserves the value; the canonical name wins when both names are supplied. |

`compile_mode="unstructured_in_simt"` is currently accepted as an equivalent compatibility spelling of `compile_mode="simd_simt_template"` and is normalized to the canonical value; it does not currently emit a `FutureWarning`. New code should use `simd_simt_template`.

#### Compiler Options to Remove from User Configuration

The following deprecated options no longer have an effective public control with the same name. When explicitly supplied during the compatibility period, they emit a `FutureWarning` and the user value is ignored. Remove them from `triton.Config`, autotune configurations, and kernel launch meta-parameters as described below.

| Deprecated option | Migration and current backend behavior |
|-------------------|----------------------------------------|
| `add_auto_scheduling` | Remove it; the DAG auto-scheduling switch has been removed and has no replacement. |
| `allow_fp8e4nv` | Remove it; the field had no effective consumer and has no replacement. |
| `arch` | Remove it; the target architecture is provided by the compilation target's `GPUTarget.arch`. |
| `auto_blockify_size` | Remove it; the field had no effective consumer and has no replacement. |
| `auto_tile_and_bind_subblock` | Remove it; tiling and sub-block binding are derived from Linalg IR and lock semantics. |
| `code_motion` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `compile_on_910_95` | Remove it; the target product is detected from the compilation target. |
| `disable_auto_inject_block_sync` | Remove it; block synchronization injection is managed by NPU IR. |
| `disable_size_align_for_cast` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `enable_auto_blockify` | Remove it; automatic block mapping and its safety analysis are backend-managed. |
| `enable_buffer_insert_optimization` | Remove it; DynamicCV keeps buffer insertion optimization enabled internally. |
| `enable_cce_vf_auto_sync` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `enable_cce_vf_remove_membar` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `enable_cross_if_fusion` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `enable_drop_unit_dims` | Remove it; use `enable_flatten` instead when flattening is intended. |
| `enable_linearize` | Remove it; the field had no effective consumer and has no replacement. |
| `enable_mask_fallback_conversion` | Remove it; the backend fixes mask fallback conversion to disabled. |
| `enable_nd2nz_on_vector` | Remove it; the backend fixes Vector ND2NZ conversion to disabled. |
| `enable_select_analysis` | Remove it; the backend fixes select analysis to enabled. |
| `enable_sync_block_lock` | Remove it; the field had no effective consumer and has no replacement. |
| `enable_ub_refine_opt` | Remove it; the backend fixes UB refine optimization to disabled. |
| `graph_optimize_emit_remarks` | Remove it; Graph Optimization remarks are fixed to disabled by the backend. |
| `graph_optimize_max_rewrites_per_function` | Remove it; the maximum rewrites per function is fixed to 64 by the backend. |
| `graph_optimize_rule_mask` | Remove it; the Graph Optimization rule mask is fixed to 511 by the backend. |
| `graph_optimize_ub_capacity_bytes` | Remove it; the Graph Optimization UB budget is derived from the target product. |
| `grid_num_tiles` | Remove it; the backend injects this value automatically from a static grid. |
| `has_auto_blockify_blacklist_op` | Remove it; the compiler derives the safety marker by scanning TTIR. |
| `kernel_name` | Remove it; the kernel name is derived from TTIR. |
| `llvm_version` | Remove it; the field had no effective consumer and has no replacement. |
| `mix_mode` | Remove it; mix mode is derived from Linalg IR as internal metadata. |
| `ops_reorder` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `optimize_dynamic_offset` | Remove it; the backend fixes dynamic-offset optimization to disabled. |
| `parallel_mode` | Remove it; parallel mode is derived from `compile_mode` and Linalg IR. |
| `storage_align` | Remove it; the former vendor compiler control has been removed and has no replacement. |
| `stream` | Remove it; launch streams are managed by the runtime and driver. |
| `use_bytecode` | Remove it; the bytecode pipeline is always enabled by the backend. |
| `vf_merge_level` | Remove it; the backend default VF merge level is used. |
| `warp_size` | Remove it; the Ascend backend fixes the warp size to 32. |

#### Deprecated Environment Variables

The following environment variables emit a `FutureWarning` when detected during the compatibility period, but their values no longer affect backend behavior. Unset them:

| Deprecated environment variable | Migration and current backend behavior |
|---------------------------------|----------------------------------------|
| `LLVM_ROOT` | Unset it; set `CC` to select the CPU launcher compiler when needed. |
| `MLIR_ROOT` | Unset it; use packaged MLIR tools or tools discoverable through `PATH`. |
| `TRITON_ALL_BLOCKS_PARALLEL` | Unset it; automatic block mapping is managed by backend policy. |
| `TRITON_ASCEND_ARCH` | Unset it; the target architecture is provided by the explicit compilation target's `GPUTarget.arch`. |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | Unset it; the variable had no effective consumer and has no replacement. |
| `TRITON_BACKEND` | Unset it; the Ascend backend policy is no longer selected by this environment variable. |
| `TRITON_DISABLE_FFTS` | Unset it; FFTS policy is derived from the explicit compilation target. |
| `TRITON_REGISTER_TENSOR_MSPROF` | Unset it; tensor-shape msprof registration is no longer controlled by an environment variable. |
