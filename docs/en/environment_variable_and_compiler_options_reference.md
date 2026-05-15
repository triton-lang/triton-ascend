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
|------|----------|--------|----------|----------|----------|
| **Debugging and logging**| TRITON_DEBUG | **0** or not set| Specifies whether to enable the debugging output function of Triton to print detailed debugging information during running. This is useful for troubleshooting problems in the compilation or execution phase. When this parameter is set to **1**, Triton outputs more information about the compilation, kernel generation, and execution. Some implementations may support more fine-grained debugging levels (such as 2 and 3), depending on the Triton version and implementation.| **0**: The debugging is disabled.<br>**1**: The debugging is enabled.| |
| **Debugging and logging**| MLIR_ENABLE_DUMP | **0** or not set| Specifies whether to dump the intermediate representation (IR) of all kernels before each MLIR optimization. You can set `MLIR_ENABLE_DUMP` to `kernelName` to dump the IR of a specific kernel.| **0**: Do not dump.<br>**1**: Dump the IR of all kernels.<br>*kernelName*: Dump the IR of a specific kernel.| The Triton cache may interfere with the dump. If `MLIR_ENABLE_DUMP=1` does not take effect, you can run `rm -r ~/.triton/cache` to clear the Triton cache.|
| **Debugging and logging**| LLVM_IR_ENABLE_DUMP | **0** or not set| Specifies whether to dump the IR before each LLVM IR optimization.| **0**: Do not dump.<br>**1**: Dump IRs.| |
| **Debugging and logging**| TRITON_REPRODUCER_PATH | Not set| Generates the MLIR reproduction file before each MLIR compilation phase. If a phase fails, `<reproducer_path>` saves the MLIR status before the failure.| `<reproducer_path>`: save path.| |
| **Debugging and logging**| TRITON_INTERPRET | **0** or not set| Specifies whether to use the Triton interpreter instead of the GPU for running and support inserting Python breakpoints in kernel function code.| **0**: Breakpoints are not supported.<br>**1**: Breakpoints are supported.| |
| **Debugging and logging**| TRITON_ENABLE_LLVM_DEBUG | **0** or not set| Specifies whether to pass the`-debug` parameter to LLVM and outputs a large amount of debugging information. If there is too much information, you can use `TRITON_LLVM_DEBUG_ONLY` to limit the output scope.| **0**: Pass.<br>**1**: Do not pass.| Another method to reduce output interference is as follows: Set the running program by setting `LLVM_IR_ENABLE_DUMP` to `1`, extract the IR before the target LLVM optimization channel, and run the `opt` tool of the LLVM separately. In this case, you can add `-debug-only=foo` to the command line to limit the debugging range.|
| **Debugging and logging**| TRITON_LLVM_DEBUG_ONLY | Not set| Equivalent to the `-debug-only` command line option of LLVM. This parameter can be used to limit the LLVM debugging output to a specific optimization channel or component name (defined by the `#define DEBUG_TYPE` macro in LLVM and Triton), thereby effectively reducing redundant debugging output. You can specify one or more comma-separated values, for example, `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`.| Comma-separated values: channel or component name| |
| **Debugging and logging**| USE_IR_LOC | **0** or not set| Specifies whether to include location information (such as file names and line numbers) in the generated IR. This information is helpful for debugging, but may increase the size of the generated IR. If this parameter is set to **1**, the IR is re-parsed, and the location information is mapped to the line number of the IR file with a specific extension (not the line number of the Python source file). This enables a direct mapping from the IR to the LLVM IR/PTX. When used with the performance analysis tool, this parameter can be used to implement fine-grained performance analysis on IR instructions.| **0**: No location information is included.<br>**1**: The location information is included.| |
| **Debugging and logging**| TRITON_PRINT_AUTOTUNING | **0** or not set| After the automatic optimization is complete, the optimal configuration and total time of each kernel are output.| **0**: Do not output.<br>**1**: Output.| |
| **Debugging and logging**| MLIR_ENABLE_REMARK | **0** or not set| Specifies whether to enable the output of remarks during MLIR compilation, including performance warnings in remarks.| **0**: Disabled.<br>**1**: Enabled.| |
| **Debugging and logging**| TRITON_KERNEL_DUMP | **0** or not set| Specifies whether to enable the dump function of the Triton kernel. When this function is enabled, Triton saves the generated kernel code (IR and final PTX in each compilation phase) to the specified directory.| **0**: Disabled.<br>**1**: Enabled.| |
| **Debugging and logging**| TRITON_DUMP_DIR | Current working directory or not set| Specifies the directory for storing the Triton kernel dump file, which is the directory for saving the IR and PTX when `TRITON_KERNEL_DUMP` is set to `1`.| **"path"**: save path.| |
| **Debugging and logging**| TRITON_DEVICE_PRINT | **0** or not set| If this parameter is set to `1` or `true` (`TRUE` is converted to `true`), the function of `tl.device_print` is enabled. Note: This function uses the GM buffer (the pointer of which is passed to the kernel).| **0**: Disabled.<br>**1**: The functionality of `tl.device_print` is enabled.| The maximum size of the GM buffer for each thread is 16 KB. If the buffer size exceeds 16 KB, the excess content will be discarded. The value is fixed currently and will be adjusted through an environment variable.|
| **Compilation control**| TRITON_ALWAYS_COMPILE | **0** or not set| Specifies whether Triton forcibly recompiles the kernel each time it runs, instead of using the existing cached version. By default, Triton caches the compiled kernels (based on parameters and configurations) to improve performance. If this parameter is set to **1**, Triton ignores the cache and recompiles the kernel each time it runs, which is useful for debugging or testing new compiler features.| **0**: Disabled.<br>**1**: All kernels are recompiled during each running.| |
| **Compilation control**| DISABLE_LLVM_OPT | **0** or not set| If this parameter is set to **1**, the optimization steps (LLVM optimization of **make_llir** and **make_ptx**) during LLVM compilation can be disabled. If this parameter is set to a character string, the LLVM optimization flags to be disabled are parsed. For example, if `DISABLE_LLVM_OPT` is set to `"disable-lsr"`, the loop strength optimization is disabled (this optimization may cause a performance fluctuation of up to 10% in some kernels with register pressure).| **0**: The LLVM optimization is enabled.<br>**1**: The optimization steps (LLVM optimization of make_llir and make_ptx) during LLVM compilation are disabled.| |
| **Compilation control**| MLIR_ENABLE_TIMING | **0** or not set| Specifies whether to enable the time statistics function during MLIR compilation.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| LLVM_ENABLE_TIMING | **0** or not set| Specifies whether to enable the time statistics function during LLVM compilation.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_DEFAULT_FP_FUSION | **1** (enabled)| Specifies whether to enable the floating-point operation fusion optimization by default. The default floating-point operation fusion behavior (for example, **mul+add->fma**) is overwritten.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_KERNEL_OVERRIDE | **0** or not set| Specifies whether to enable the Triton kernel override function. You can use the user-specified external file (such as IR/PTX) to override the default generated kernel code at the beginning of each compilation phase.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_OVERRIDE_DIR | Current working directory or not set| Specifies the directory for searching the Triton kernel override file. Directory for loading the IR/PTX file when `TRITON_KERNEL_OVERRIDE` is set to `1`.| **"path"**: save path.| |
| **Compilation control**| TRITON_ASCEND_COMPILE_SPEED_OPT | **0** or not set| Specifies whether the JIT compiler skips the subsequent compilation phase after detecting that the kernel compilation fails. Set the parameter to `1` to skip the attempt. (The default value `0` indicates that the attempt is continued.)| **0**: Continue the attempt.<br>**1**: Skip.| |
| **Compilation control**| TRITON_COMPILE_ONLY | **0** or not set| Specifies whether to perform only compilation without execution. This parameter is used when **remote_launch** is used.| **0**: Disabled.<br>**1**: Enabled.| |
| **Compilation control**| TRITON_DISABLE_FFTS | **0** or not set| Specifies whether to disable FFTS.| **0**: Enabled.<br>**1**: Disabled.| |
| **Running and scheduling**| TRITON_ALL_BLOCKS_PARALLEL | **0** or not set| Specifies whether to enable the automatic optimization of the number of logical cores based on the number of physical cores. This parameter can be enabled only when logical cores can execute in parallel. When the number of logical cores is greater than the number of physical cores, enabling this parameter will instruct the compiler to automatically adjust the number of logical cores to match the number of physical cores, thereby reducing scheduling overhead. After this parameter is enabled, the value of **grid** can be greater than 65535. Limitation: This option can be enabled only when the logic of the Triton kernel is insensitive to the execution sequence. Otherwise, a deadlock may occur. The per-kernel option `enable_auto_blockify` (see `architecture_difference.md`) takes precedence over this env var when set; the env var only acts as the default for kernels that leave `enable_auto_blockify` unset.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling**| TRITON_ENABLE_TASKQUEUE | **0** or not set| Specifies whether to enable **task_queue**.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling**| TRITON_ENABLE_SANITIZER | **0** or not set| Specifies whether to enable SANITIZER.| **0**: Disabled.<br>**1**: Enabled.| |
| **Running and scheduling**| ENABLE_PRINT_UB_BITS | **0** or not set| After this parameter is enabled, the current UB usage can be obtained for the inductor.| **0**: Disabled.<br>**1**: Enabled.| |
| **Others**| TRITON_BENCH_METHOD | Not set| When the Ascend NPU is used, change `do_bench` in `testing.py` to `do_bench_npu`. (This parameter is used when `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE` is set to `1`.) If this parameter is set to `default`, the original `do_bench` function is still called even if the NPU is available.| **"npu"**: Switch to `do_bench_npu`.| |
| **Others**| TRITON_REMOTE_RUN_CONFIG_PATH | path | Specifies the configuration path for remote running.| Specify the path directly.| |

## Compiler Options

Compiler options control the compilation strategy for a single Triton kernel and can be passed through `triton.Config`, autotune parameters, or kernel launch meta-parameters.

### Compiler Option Usage Example

For example, pass `multibuffer` directly during kernel launch:

```python
import triton
import triton.language as tl

@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...

grid = (triton.cdiv(n_elements, 1024),)
kernel[grid](..., BLOCK_SIZE=1024, multibuffer=True)
```

### Compiler Option Reference Table

The following table describes the options.

| Category | Compiler Option | Default/Values | Function Description | Setting Description |
|----------|-----------------|----------------|----------------------|--------------------|
| **General pipeline** | `multibuffer` | `True`, `False`; disabled by default in 910_95 compilation scenarios | Enables or disables ping-pong/double-buffer pipelines. | `triton.Config` or launch meta-parameter |
| **CV fusion** | `enable_auto_bind_sub_block` | `None`, `True`, `False` | Enables or disables automatic sub-block binding. | `triton.Config` or launch meta-parameter |
| **CV fusion** | `enable_hivm_auto_cv_balance` | `None`, `True`, `False` | Enables or disables automatic CV balance. | `triton.Config` or autotune parameter |
| **CV fusion/sync** | `sync_solver` | `None`, `True`, `False` | Enables or disables the HIVM synchronization solver. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `unit_flag` | `None`, `True`, `False` | Cube-output synchronization option. | `triton.Config` or autotune parameter |
| **Synchronization** | `inject_barrier_all` | `None`, `True`, `False` | Enables or disables automatic barrier synchronization injection. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `inject_block_all` | `None`, `True`, `False` | Enables or disables automatic block synchronization injection. | `triton.Config` or launch meta-parameter |
| **Multibuffer scope** | `limit_auto_multi_buffer_only_for_local_buffer` | `None`, `True`, `False` | Restricts automatic multi-buffering to local buffers. | `triton.Config` or autotune parameter |
| **Multibuffer scope** | `limit_auto_multi_buffer_of_local_buffer` | `None`, `"no-limit"`, `"no-l0c"` | Configures the local-buffer automatic multi-buffering scope. | `triton.Config` or autotune parameter |
| **Workspace** | `set_workspace_multibuffer` | `None`, `2`, `4` | Configures workspace multi-buffering. | `triton.Config` or autotune parameter |
| **CV fusion tiling** | `tile_mix_vector_loop` | `None`, `2`, `4`, `8` | Configures the Vector loop split count. | `triton.Config` or autotune parameter |
| **CV fusion tiling** | `tile_mix_cube_loop` | `None`, `2`, `4`, `8` | Configures the Cube loop split count. | `triton.Config` or autotune parameter |
| **CV fusion/sync** | `disable_auto_inject_block_sync` | `None`, `True`, `False` | Enables or disables automatic block sync injection. | `triton.Config` or launch meta-parameter |
| **Runtime stream** | `stream` | `None` or NPU stream identifier | Specifies the NPU stream. | launch meta-parameter |
| **Compiler pass** | `enable_linearize` | Version-dependent | Enables or disables the linearization pass. | `triton.Config` or launch meta-parameter |
| **CV fusion/layout** | `enable_nd2nz_on_vector` | Default `False` | Enables or disables ND-to-NZ layout transformation on the Vector path. | `triton.Config` or launch meta-parameter |
| **Large-grid optimization** | `auto_blockify_size` | Default `1` | Enables or disables AutoBlockify pass. Ignored when `TRITON_ALL_BLOCKS_PARALLEL` is not set. | launch meta-parameter or `triton.Config` |
