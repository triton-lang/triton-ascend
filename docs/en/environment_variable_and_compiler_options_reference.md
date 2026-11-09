# Environment Variables and Compiler Options

This page is a configuration reference for Triton-Ascend kernel developers. It lists settings for compilation, execution, debugging, and autotuning, with their purpose, defaults, and support on A2/A3 and 950. Environment variables configure compilation and runtime behavior within a process; compiler options configure individual kernels. Environment variables can also be read and written in-process through `triton.knobs` (including the Ascend-specific `triton.knobs.ascend` group), without editing startup scripts. Migration advice for existing configurations appears under [Compiler Option Cleanup and Compatibility](#compiler-option-cleanup-and-compatibility).

## Environment Variables

### Environment Variable Usage Example

Use `export` to set environment variables before starting the Python program. For example, set `TRITON_DEBUG=1` to enable debugging:

```bash
export TRITON_DEBUG=1
python3 run_kernel.py
```

Programs subsequently launched from the current terminal inherit variables set with `export`. To set a variable for a single invocation, place the assignment before the command:

```bash
TRITON_DEBUG=1 python3 run_kernel.py
```

### Environment Variable Reference Table

The 950 column covers Ascend 950PR/950DT; "—" means not applicable. Compilation mode restrictions are defined before the [compiler option table](#compiler-options-reference). Unless noted otherwise, boolean environment variables accept `1` to enable them.

| Category | Environment variable | Description | Default | A2/A3 | 950 |
|----------|----------------------|-------------|---------|-------|-----|
| Debugging | `TRITON_DEBUG` | Enables JIT debugging and device assertions and saves TTIR; the normal path also logs compilation and saves backend IR and binaries. | Off | Yes | Yes |
| Debugging | `USE_IR_LOC` | Rebuilds locations from an IR file. For AST compilation, `ttir` supports this; IR inputs also support source-file location snapshots. | Unset | Yes | Yes |
| Debugging | `TRITON_FRONT_END_DEBUGGING` | Preserves the original Python traceback for frontend errors. | Off | Yes | Yes |
| Debugging | `TRITON_KERNEL_DUMP` | Saves compilation artifacts and the final `npubin` when a compilation runs. | Off | Yes | Yes |
| Debugging | `TRITON_DUMP_DIR` | Sets the output directory for `TRITON_KERNEL_DUMP`. | `$TRITON_HOME/.triton/dump` | Yes | Yes |
| Debugging | `TRITON_DISABLE_LINE_INFO` | `true/1` omits line information from the BiSheng request; `false/0` enables it. | `true` | Yes | Normal path |
| Debugging | `LLVM_EXTRACT_DI_LOCAL_VARIABLES` | Requests debug-variable information from the BiSheng compiler. | Off | Yes | Normal path |
| Debugging | `TRITON_DEVICE_PRINT` | Enables the device GM buffer used by `tl.device_print`. | Off | Yes | Yes |
| Debugging | `TRITON_MEMORY_DISPLAY` | Requests compiler memory-usage output. | Off | Yes | — |
| Debugging | `ENABLE_PRINT_UB_BITS` | Requests the number of occupied UB bits from the compiler. | Off | Yes | Normal path |
| Debugging | `TRITON_INTERPRET` | Runs supported Triton operations in the CPU interpreter for Python breakpoint debugging. | Off | Yes | Yes |
| Debugging | `TRITON_ENABLE_SANITIZER` | Passes the sanitizer switch to the Ascend NPU compiler. | Off | Yes | Normal path |
| Compilation | `TRITON_ALWAYS_COMPILE` | Ignores compilation cache hits and recompiles kernels. | Off | Yes | Yes |
| Compilation | `TRITON_KERNEL_OVERRIDE` | Loads an external IR or binary with a matching stage filename. | Off | Yes | Yes |
| Compilation | `TRITON_OVERRIDE_DIR` | Sets the directory containing override files. | `$TRITON_HOME/.triton/override` | Yes | Yes |
| Compilation | `TRITON_F32_DEFAULT` | Sets the default F32 input precision for `tl.dot`; Ascend uses `ieee` or `hf32`, and maps compatibility input `tf32` to `hf32`. | Unset; backend uses `ieee` | Yes | Yes |
| Compilation | `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | Allows non-`constexpr` globals in the JIT frontend. | Off | Yes | Yes |
| Compilation | `TRITON_ENABLE_LIBDEVICE` | Adds a libdevice link on the A2/A3 compiler path. Any nonempty value, including `"0"`, enables it. | Unset | Yes | — |
| Compilation | `TRITON_ENABLE_LIBDEVICE_SIMT` | Selects SIMT libdevice extern symbols on Ascend 950. Any nonempty value, including `"0"`, enables it. | Unset | — | Yes |
| Cache and autotuning | `TRITON_HOME` | Sets the base for cache, dump, and override directories. | User home | Yes | Yes |
| Cache and autotuning | `TRITON_CACHE_DIR` | Sets the compilation cache directory. | `$TRITON_HOME/.triton/cache` | Yes | Yes |
| Cache and autotuning | `TRITON_CACHE_AUTOTUNING` | Caches autotune selections. | Off | Yes | Yes |
| Cache and autotuning | `TRITON_PRINT_AUTOTUNING` | Set to `1` to print autotune results and normal-path compiler commands. | Off | Yes | Yes |
| Cache and autotuning | `TRITON_CACHE_MANAGER` | Selects a compilation cache manager with `MODULE:CLASS`; use `triton.runtime.cache:RemoteCacheManager` for remote caching. | Local cache manager | Yes | Yes |
| Cache and autotuning | `TRITON_REMOTE_CACHE_BACKEND` | Selects the remote cache storage backend with `MODULE:CLASS`; requires the remote cache manager. | Unset | Yes | Yes |
| Cache and autotuning | `TRITON_ENABLE_UBTUNER` | Selects the UB tuning mode, `compile` or `run`. | Unset | Yes | Normal path |
| Cache and autotuning | `TRITON_AUTOTUNE_PARALLEL_COMPILE` | Controls parallel compilation in Ascend autotune; `0` disables it. | `1` | Yes | Yes |
| Cache and autotuning | `TRITON_BENCH_METHOD` | `npu` selects Ascend's `do_bench_npu`; `default` uses the shared benchmark path. | `default` | Yes | Yes |
| Runtime | `TRITON_COMPILE_ONLY` | Set to `1/true` to skip kernel launches; compilation and launcher extension loading still run. | Off | Yes | Yes |
| Runtime | `TRITON_ENABLE_TASKQUEUE` | Controls the Ascend launcher's task queue. | On | Yes | Yes |
| Runtime | `TRITON_GRID_WARN_PRINT` | Prints Ascend launcher warnings about grid size. | Off | Yes | Yes |
| Runtime | `NPU_DEVICE_LIMIT` | Caps AI Core and Vector Core counts with `AIcores,VectorCores`, for example `14,28`; the device ratio must be valid. | Unset; device counts | Yes | Yes |
| Toolchain selection | `CC` | Sets the C/C++ compiler for the CPU launcher extension. | Unset; system default compiler | Yes | Yes |
| Toolchain selection | `TRITON_NPU_COMPILER_PATH` | Supplies a fallback NPU compiler tool directory when packaged tools and `PATH` lookup fail. | Unset | Yes | Yes |
| Toolchain selection | `TRITON_BUILD_DIR` | Searches this directory's `bin/` for `triton-opt` and `triton-mlir-opt`, after packaged tools and before `PATH`. | Unset | Yes | Yes |

### triton.knobs Interface

`triton.knobs` is the typed configuration interface introduced in upstream Triton 3.4: descriptor fields correspond to environment variables, and a few entries are Python objects (such as callbacks) that have no corresponding environment variable and can only be set in-process. If you only need to configure a setting once before startup, set the environment variable directly. Use `triton.knobs` when you need to read the current value in-process, change a setting temporarily, or restore it by scope. Upstream groups are provided directly by `triton.knobs`; Triton-Ascend adds the `triton.knobs.ascend` group in Triton 3.6. It becomes available once the Ascend backend is imported.

#### Quick start

```python
import triton
import triton.backends.ascend.runtime  # registers the Ascend backend and installs triton.knobs.ascend

# Show all fields and current values in a group
print(triton.knobs.ascend.knobs)

# Read and assign a field
print(triton.knobs.compilation.dump_ir)
triton.knobs.ascend.device_print = True
```

Assignment affects subsequent reads and, by default, writes the environment variable too. See the read and write rules below.

#### Read and write rules

- Read: returns the in-process value first, then the environment variable, then the default.
- Assign: stores an in-process value and, by default, writes the environment variable too.
- `scope()`: records the in-process values and environment variables on entry and restores both on exit, including on exceptions. This is the recommended way to switch settings temporarily.
- `reset()` / `knobs.env`: clears in-process overrides only; environment variables already written are not restored.
- `copy()`: copies the current in-process overrides and returns an independent group object.
- `propagate_env = False`: assignment affects the current Python process only; subprocesses and tools that call `os.getenv` directly cannot see it. Use it only when no external reader depends on the variable.

For example, to enable `device_print` temporarily and restore it after the block:

```python
with triton.knobs.ascend.scope():
    triton.knobs.ascend.device_print = True
    kernel[grid](x, y, out)
# Restored after leaving the block
```

#### Module-level fields

`propagate_env` applies to `triton.knobs` as a whole, belongs to no group, and has no corresponding environment variable; it can only be set through Python assignment:

```python
triton.knobs.propagate_env = False  # subsequent assignments are no longer written to os.environ
```

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `propagate_env` | — | `True` | `True` / `False` | Controls whether knob assignments are written back to `os.environ`; when `False`, assignments affect only the current process. |

#### Groups

The table below lists each configuration group by purpose; the fields of each group are described in the following sections. If you are interested only in Ascend-specific switches, refer directly to the `triton.knobs.ascend` group.

| Group | Purpose | Ascend support |
|-------|---------|----------------|
| [`triton.knobs.ascend`](#knobs-ascend) | Ascend-specific compilation, autotuning, and launcher switches | Yes |
| [`triton.knobs.compilation`](#knobs-compilation) | Cache override, IR dump, debug information, and other compilation behavior | Some fields apply |
| [`triton.knobs.runtime`](#knobs-runtime) | Interpreter mode and runtime switches | Some fields apply |
| [`triton.knobs.cache`](#knobs-cache) | Cache directories and cache managers | Yes |
| [`triton.knobs.autotuning`](#knobs-autotuning) | Autotune result caching and printing | Yes |
| [`triton.knobs.language`](#knobs-language) | Language-level defaults such as FP32 precision | Yes |
| [`triton.knobs.build`](#knobs-build) | Build tools and a custom build entry point for the CPU launcher extension | `cc` and `impl` only |
| [`triton.knobs.redis`](#knobs-redis) | Redis storage backend for remote caching | Yes |
| `triton.knobs.nvidia` / `.amd` / `.proton` | Other backend-specific settings | Not used |

<a id="knobs-ascend"></a>

**`triton.knobs.ascend`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `enable_sanitizer` | `TRITON_ENABLE_SANITIZER` | `False` | `True` / `False` | Passes the sanitizer switch to the Ascend NPU compiler. |
| `print_ub_bits` | `ENABLE_PRINT_UB_BITS` | `False` | `True` / `False` | Requests the number of occupied UB bits from the compiler. |
| `memory_display` | `TRITON_MEMORY_DISPLAY` | `False` | `True` / `False` | Requests compiler memory-usage output. |
| `npu_compiler_path` | `TRITON_NPU_COMPILER_PATH` | `None` | Directory path / `None` | Fallback NPU compiler tool directory used only when packaged tools and `PATH` lookup fail. |
| `enable_ubtuner` | `TRITON_ENABLE_UBTUNER` | `""` | `""` (off), `"compile"`, `"run"` | Selects the UB tuning mode. |
| `autotune_parallel_compile` | `TRITON_AUTOTUNE_PARALLEL_COMPILE` | `True` | `True` / `False` | Controls parallel compilation in Ascend autotune. |
| `bench_method` | `TRITON_BENCH_METHOD` | `"default"` | `"default"`, `"npu"` | `npu` selects Ascend's `do_bench_npu`; `default` uses the shared benchmark path. |
| `compile_only` | `TRITON_COMPILE_ONLY` | `False` | `True` / `False` | Skips kernel launches; compilation and launcher extension loading still run. |
| `enable_taskqueue` | `TRITON_ENABLE_TASKQUEUE` | `True` | `True` / `False` | Controls the Ascend launcher's task queue. |
| `device_print` | `TRITON_DEVICE_PRINT` | `False` | `True` / `False` | Enables the device GM buffer used by `tl.device_print`. |
| `grid_warn_print` | `TRITON_GRID_WARN_PRINT` | `False` | `True` / `False` | Prints Ascend launcher warnings about grid size. |

<a id="knobs-compilation"></a>

**`triton.knobs.compilation`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `override` | `TRITON_KERNEL_OVERRIDE` | `False` | `True` / `False` | Loads an external IR or binary with a matching stage filename from the override directory. |
| `dump_ir` | `TRITON_KERNEL_DUMP` | `False` | `True` / `False` | Saves compilation-stage artifacts and the final `npubin`; only applies to compilations that actually run. |
| `dump_ir_extract_di_local_variables` | `LLVM_EXTRACT_DI_LOCAL_VARIABLES` | `False` | `True` / `False` | Requests debug-variable information from the BiSheng compiler. |
| `store_binary_only` | `TRITON_STORE_BINARY_ONLY` | `False` | `True` / `False` | Writes only `cubin`/`hsaco`/`json` artifacts, not intermediate IR; the Ascend binary extension is `npubin`, which is not in that list. |
| `always_compile` | `TRITON_ALWAYS_COMPILE` | `False` | `True` / `False` | Ignores compilation cache hits and recompiles kernels. |
| `use_ir_loc` | `USE_IR_LOC` | `None` | `None`, a stage name (such as `"ttir"`), or a source-location snapshot | Rebuilds locations from the line numbers of the specified stage's IR file. |
| `enable_asan` | `TRITON_ENABLE_ASAN` | `False` | `True` / `False` | Not supported on Ascend. |
| `disable_line_info` | `TRITON_DISABLE_LINE_INFO` | `False` | `True` / `False` | Omits line information from the BiSheng request. The Ascend backend reads this variable directly and treats it as `True` when unset. |
| `front_end_debugging` | `TRITON_FRONT_END_DEBUGGING` | `False` | `True` / `False` | Preserves the original Python traceback for frontend errors. |
| `allow_non_constexpr_globals` | `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | `False` | `True` / `False` | Allows non-`constexpr` globals in the JIT frontend. |
| `instrumentation_mode` | `TRITON_INSTRUMENTATION_MODE` | `""` | `""` (off), `"consan"` | Not supported on Ascend. |
| `listener` | — | `None` | `CompilationListener` / `None` | Called after compilation (including cache hits) with `src`, `metadata`, `metadata_group`, `times`, and `cache_hit`; useful for compile-time statistics. |

<a id="knobs-runtime"></a>

**`triton.knobs.runtime`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `interpret` | `TRITON_INTERPRET` | `False` | `True` / `False` | Runs supported Triton operations in the CPU interpreter. |
| `override_arch` | `TRITON_OVERRIDE_ARCH` | `None` | Architecture string / `None` | Not supported on Ascend. |
| `debug` | `TRITON_DEBUG` | `False` | `True` / `False` | Enables JIT debugging and device assertions; the environment variable is read once at module import. |
| `launch_enter_hook` | — | `HookChain()` | `HookChain[LaunchHook]` | Called before kernel launch with the launch metadata; add or remove callbacks with `.add(func)` / `.remove(func)`. |
| `launch_exit_hook` | — | `HookChain(reversed=True)` | `HookChain[LaunchHook]` | Called after kernel launch; the chain runs in reverse registration order. |
| `kernel_load_start_hook` | — | `HookChain()` | `HookChain[InitHandleHook]` | Called before the kernel binary/module is loaded; `module` and `function` may be `None` at this point. |
| `kernel_load_end_hook` | — | `HookChain(reversed=True)` | `HookChain[InitHandleHook]` | Called after loading completes; the chain runs in reverse registration order. |
| `jit_cache_hook` | — | `None` | `JITHook` / `None` | Called before each JIT compilation with the key, signature, and compile options; a truthy return skips this compilation. |
| `jit_post_compile_hook` | — | `None` | `JITHook` / `None` | Called after the JIT compilation call returns; `jit_cache_hook` for the same kernel always runs before it. |
| `add_stages_inspection_hook` | — | `None` | `PipelineStagesHook` / `None` | Not supported on Ascend. |

<a id="knobs-cache"></a>

**`triton.knobs.cache`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `home_dir` | `TRITON_HOME` | User home | Directory path | Base for the Triton cache, dump, and override directories. |
| `dump_dir` | `TRITON_DUMP_DIR` | `$TRITON_HOME/.triton/dump` | Directory path | Output directory for IR dumps. |
| `override_dir` | `TRITON_OVERRIDE_DIR` | `$TRITON_HOME/.triton/override` | Directory path | Directory containing override files. |
| `dir` | `TRITON_CACHE_DIR` | `$TRITON_HOME/.triton/cache` | Directory path | Compilation cache directory. |
| `manager_class` | `TRITON_CACHE_MANAGER` | `None` | `MODULE:CLASS` / `None` | Compilation cache manager; use `triton.runtime.cache:RemoteCacheManager` for remote caching. |
| `remote_manager_class` | `TRITON_REMOTE_CACHE_BACKEND` | `None` | `MODULE:CLASS` / `None` | Remote cache storage backend; requires the remote cache manager. |

<a id="knobs-autotuning"></a>

**`triton.knobs.autotuning`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `cache` | `TRITON_CACHE_AUTOTUNING` | `False` | `True` / `False` | Caches autotune selections. |
| `print` | `TRITON_PRINT_AUTOTUNING` | `False` | `True` / `False` | Prints autotune results and normal-path compiler commands. |

<a id="knobs-language"></a>

**`triton.knobs.language`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `fp32_default` | `TRITON_F32_DEFAULT` | `None` | `None`, `"ieee"`, `"hf32"`, `"tf32"` (mapped to `hf32`) | Default FP32 input precision for `tl.dot`; `None` uses the backend default `ieee`. |
| `default_fp_fusion` | `TRITON_DEFAULT_FP_FUSION` | `True` | `True` / `False` | Whether floating-point operation fusion is allowed by default. |

<a id="knobs-build"></a>

**`triton.knobs.build`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `cc` | `CC` | `None` | Compiler path or name / `None` | C/C++ compiler for the CPU launcher extension. |
| `cudacrt_path` | `TRITON_CUDACRT_PATH` | `None` | Directory path / `None` | Not supported on Ascend. |
| `cudart_path` | `TRITON_CUDART_PATH` | `None` | Directory path / `None` | Not supported on Ascend. |
| `impl` | — | `None` | `BuildImpl` / `None` | Replaces the default local build path for the CPU launcher extension; signature is `(name, src, srcdir, library_dirs, include_dirs, libraries)`. |

<a id="knobs-redis"></a>

**`triton.knobs.redis`**

| Field | Environment variable | Default | Value range | Description |
|-------|----------------------|---------|-------------|-------------|
| `key_format` | `TRITON_REDIS_KEY_FORMAT` | `"triton:{key}:{filename}"` | Format string containing `{key}` and `{filename}` | Remote cache key format. |
| `host` | `TRITON_REDIS_HOST` | `"localhost"` | Hostname or IP | Redis address. |
| `port` | `TRITON_REDIS_PORT` | `6379` | Integer, valid port `1`–`65535` | Redis port. |

## Compiler Options

### Compiler Option Usage Example

Pass compiler options directly to `kernel[grid](...)`, or use `triton.Config` to define autotuning candidates. Use the separate `triton.Config` constructor parameters for `num_warps`, `num_stages`, and `ir_override`; put other supported compiler options in `kwargs`.

In this example, `BLOCK_SIZE` is a kernel-defined `tl.constexpr` parameter; `compile_mode` and `num_stages` are compiler options:

```python
import triton

kernel[grid](*args, BLOCK_SIZE=1024, compile_mode="simd", num_stages=1)

configs = [
    triton.Config({"BLOCK_SIZE": 1024, "compile_mode": "simd"}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024, "compile_mode": "simd"}, num_stages=2),
]
```

See the [Autotune Guide](./autotune_guide.md) for how to use the candidate configurations.

<a id="compiler-options-reference"></a>

### Compiler Option Reference Table

The table lists Ascend backend defaults and notes any different `triton.Config` defaults. `None` usually leaves the decision to the backend or toolchain.

- **Normal path**: `compile_mode="simd"` or `"simd_simt_template"`.
- **Pure SIMT only**: `compile_mode="simt_only"`, available only on 950.
- **Requires active DynamicCV**: the compiler has successfully applied [DynamicCV pipelining](./architecture_design_and_core_features.md#dynamic-cv-pipeline), meaning `enable_dynamic_cv_pipeline=True` is enabled on the 950 normal path and no fallback has occurred.

| Category | Compiler option | Description | Default / values | A2/A3 | 950 |
|----------|-----------------|-------------|------------------|-------|-----|
| Shared Triton | `debug` | Enables device assertions and compilation debugging. Under JIT, enabling either this option or `TRITON_DEBUG` enables debugging. | `False`; boolean | Yes | Yes |
| Shared Triton | `sanitize_overflow` | Enables integer-overflow checks implemented with device assertions; also requires `debug`. | `True`; boolean | Yes | Yes |
| Shared Triton | `num_stages` | Setting `1` disables automatic multi-buffering; other values use automatic mode without specifying a buffer count per level. | When `multibuffer_mode` is unset: backend `2`, `triton.Config` `3`; integer | Yes | Normal path |
| Shared Triton | `num_warps` | Sets the warp count for pure-SIMT compilation. | Backend `32`; `triton.Config` `4`; positive integer | — | Pure SIMT only |
| Shared Triton | `ir_override` | Replaces a compilation-stage artifact with a file of the same extension. | `None`; file path | Yes | Yes |
| Compilation mode and graph optimization | `compile_mode` | `simd` uses SIMD; `simd_simt_template` uses SIMD on A2/A3 and enables template-SIMT subpaths on 950; `simt_only` uses pure SIMT. | `"simd_simt_template"`; also `"simd"`, `"simt_only"` | First two | All three |
| Compilation mode and graph optimization | `enable_graph_optimize` | Controls TTIR Graph Optimization. | `True`; boolean | Yes | Yes |
| Multibuffering | `multibuffer` | Controls the automatic multi-buffering master switch; `num_stages=1` also turns it off. | `True`; boolean | Yes | Normal path |
| Multibuffering | `multibuffer_mode` | Sets multi-buffer slot counts for `gm`, `l1`, `l0c`, and `ub`; `1` disables multi-buffering at that level, and values greater than `1` specify the slot count. Controlled by `multibuffer`; cannot be explicitly combined with `num_stages`. | `None`; a dictionary containing all four levels, with positive signed 32-bit integer values | Yes | Normal path |
| Multibuffering | `set_workspace_multibuffer` | Sets workspace multi-buffering level. With explicit `multibuffer_mode` and `gm>1`, NPU IR overrides this value with the GM slot count. | `None`; compiler-defined integer | Yes | Normal path |
| Synchronization | `sync_solver` | Controls the HIVM graph sync solver. | `None`; boolean | Yes | Normal path |
| Synchronization | `unit_flag` | Controls HIVM unit-flag synchronization. | `None`; boolean | Yes | Normal path |
| Synchronization | `inject_barrier_all` | Controls automatic barrier injection. | `None`; boolean | Yes | Normal path |
| Synchronization | `inject_block_all` | Controls automatic block synchronization injection. | `None`; boolean | Yes | Normal path |
| Synchronization | `disable_auto_inject_block_sync` | Disables automatic inter-core synchronization insertion. Set `disable_auto_inject_block_sync=True` when using any of the Ascend affinity APIs `sync_block_all`, `sync_block_set`, or `sync_block_wait`. | `None`; boolean | Yes | Normal path |
| Scheduling and pipelines | `enable_auto_bind_sub_block` | Overrides backend-derived automatic sub-block binding. | `None`; boolean | Yes | Normal path |
| Scheduling and pipelines | `enable_hivm_auto_cv_balance` | Controls automatic HIVM CV balancing. | `None`; boolean | Yes | Normal path |
| Scheduling and pipelines | `tile_mix_vector_loop` | Sets the Vector loop split count. | `None`; integer | Yes | — |
| Scheduling and pipelines | `tile_mix_cube_loop` | Sets the Cube loop split count. | `None`; integer | Yes | — |
| Scheduling and pipelines | `enable_mixed_cv` | Controls the NPU compiler's mixed-CV path. | `None`; boolean | — | Normal path |
| Scheduling and pipelines | `enable_dynamic_cv_pipeline` | Enables [DynamicCV pipelining](./architecture_design_and_core_features.md#dynamic-cv-pipeline) on 950. When applied, it sets `set_workspace_multibuffer=0`, `enable_mixed_cv=True`, and `disable_auto_inject_block_sync=True`. If the optimization cannot be applied, the compiler restores these options to their original values and continues compilation. | `None`, resolved by target: A2/A3 `False`, 950 `True`; boolean | — | Normal path |
| Scheduling and pipelines | `buf_slot_num_of_veccore` | Sets Vector-core-local buffer slots. | `None`; integer | — | Requires active DynamicCV |
| Scheduling and pipelines | `buf_slot_num_of_crosscore` | Sets cross-core buffer slots. | `None`; integer | — | Requires active DynamicCV |
| Scheduling and pipelines | `buf_slot_num_of_gm` | Sets GM load buffer slots. | `None`; integer | — | Requires active DynamicCV |
| Memory optimization | `enable_ubuf_saving` | Controls UB saving. | `None`; boolean | Yes | — |
| Memory optimization | `disable_size_align_for_cast` | Controls cast size alignment. | `None`; boolean | Yes | — |
| Memory optimization | `disable_tightly_coupled_buffer_reuse` | Disables tightly coupled buffer reuse in the NPU compiler. | `False`; boolean | — | Normal path |
| Memory optimization | `plan_memory_strategy` | Selects a memory-planning strategy. | `None`; `"default"`, `"largest-first"` | — | Normal path |
| Operator optimization | `enable_preload` | Controls NPU compiler preload optimization. | `None`; boolean | Yes | Normal path |
| Operator optimization | `enable_flatten` | Controls the NPU compiler flatten pass. | `None`; boolean | Yes | Normal path |
| Operator optimization | `enable_auto_vectorize_v2` | Controls NPU compiler AutoVectorizeV2. | `None`; boolean | Yes | Normal path |
| Operator optimization | `vf_fusion_mode` | Selects the VF fusion strategy. | `None`, switch omitted; `"max-parallel"`, `"all-op"`, `"ub-aware-op"` | — | Normal path |
| Operator optimization | `enable_vf_stack_limit` | Limits fusion based on vector function (VF) stack-usage estimates during BiShengIR compilation to reduce the risk of stack overflow. | `False`; boolean | — | Normal path |
| Operator optimization | `hfusion_enable_multiple_consumer_fusion` | Controls multiple-consumer HFusion. | `None`; boolean | — | Normal path |
| Operator optimization | `enable_hivm_batch_matmul` | Enables batch matmul in BiShengIR compilation. | `False`; boolean | — | Normal path |
| Operator optimization | `auto_vectorize_v2_max_fused_ops_num` | Sets the AutoVectorizeV2 fused-op limit. | `None`; integer | — | Normal path |
| Operator optimization | `prevec_max_fused_ops_num` | Sets the pre-vectorization fused-op limit. | `None`; integer | — | Normal path |
| Pure SIMT | `simt_optimization_mode` | Sets the pure-SIMT multi-digit optimization mode. Explicit values on A2/A3 or the 950 normal path emit a `UserWarning` and are ignored. | `0`; toolchain-defined integer | — | Pure SIMT only |
| Pure SIMT | `shared_mem_dynamic_size` | Sets pure-SIMT dynamic shared memory size. | `122880` bytes; integer byte count | — | Pure SIMT only |
| Pure SIMT | `disable_fma` | Disables FMA optimization on the pure-SIMT path. | `False`; boolean | — | Pure SIMT only |
| Pure SIMT | `superblock_factor` | Sets the pure-SIMT superblock factor when automatic block mapping is active and row coalescing has not applied. | `1`; passed when greater than `1` | — | Pure SIMT only |

<a id="compiler-option-cleanup-and-compatibility"></a>

## Compiler Option Cleanup and Compatibility

The legacy compiler options listed below are accepted only during the compatibility period. **Once these options are fully removed in a future release, passing them through kernel launch arguments or `triton.Config` will cause an error.** Migrate or remove them as described below before upgrading.

Renamed and deprecated settings handled by Triton-Ascend emit a `FutureWarning`, once per name per process. NPU IR emits the deprecation diagnostics for legacy multibuffer options.

### Migrating Legacy Multibuffer Options

NPU IR deprecates the following three options. Their legacy behavior remains effective during the compatibility period; when `multibuffer_mode` is also set, their values are ignored. Use the equivalent configuration suggested by the compiler and specify all four levels in the dictionary:

| Legacy option | Corresponding setting in `multibuffer_mode` |
|---------------|--------------------------------------------|
| `limit_auto_multi_buffer_only_for_local_buffer` | `True` corresponds to `gm=1`, disabling GM multi-buffering. |
| `limit_auto_multi_buffer_of_local_buffer` | `"no-l0c"` corresponds to `l0c=1`; `"no-limit"` corresponds to `l0c>1`. |
| `limit_auto_multi_buffer_buffer` | On the 950 normal path: `"only-cube"` corresponds to `ub=1`; `"only-vector"` corresponds to `l1=1` and `l0c=1`; `"no-limit"` corresponds to `l1>1` and `ub>1`. |

For example, to enable double-buffering only for L1:

```python
triton.Config({"multibuffer_mode": {"gm": 1, "l1": 2, "l0c": 1, "ub": 1}})
```

### Renamed Compiler Options

Replace the following old names with the corresponding names or values:

| Deprecated name | Canonical name or value | Compatibility behavior |
|-----------------|-------------------------|------------------------|
| `force_simt_only` | `compile_mode="simt_only"` | `True` overrides an existing `compile_mode`; `False` leaves the mode unchanged. This option takes precedence when both legacy force switches are `True`. |
| `force_simt_template` | `compile_mode="simd_simt_template"` | `True` overrides an existing `compile_mode`; `False` leaves the mode unchanged. |
| `intra_cache_num` | `buf_slot_num_of_veccore` | Preserves the value; the new name wins when both names are supplied. |
| `inter_cache_num` | `buf_slot_num_of_crosscore` | Preserves the value; the new name wins when both names are supplied. |
| `load_cache_num` | `buf_slot_num_of_gm` | Preserves the value; the new name wins when both names are supplied. |
| `enable_bishengir_simt_optimization` | `simt_optimization_mode` | Preserves the value; the new name wins when both names are supplied. Effective only on the 950 pure-SIMT path. |

The compatibility value `compile_mode="unstructured_in_simt"` is equivalent to `compile_mode="simd_simt_template"` and remains accepted without a `FutureWarning`. Use `simd_simt_template` in new code.

### Deprecated Compiler Options

Remove the following options from kernel launch arguments and `triton.Config`; the compiler ignores their values.

| Deprecated option | Replacement or current behavior |
|-------------------|----------------------------------------|
| `add_auto_scheduling` | The DAG auto-scheduling switch has been removed and has no replacement. |
| `allow_fp8e4nv` | No replacement. |
| `arch` | The target architecture is provided by the compilation target's `GPUTarget.arch`. |
| `auto_blockify_size` | No replacement. |
| `auto_tile_and_bind_subblock` | The compiler determines tiling and sub-block binding automatically. |
| `bisheng_options` | User values are ignored. |
| `code_motion` | No replacement. |
| `compile_on_910_95` | The target product is detected from the compilation target. |
| `enable_auto_blockify` | The compiler manages automatic block mapping and its safety analysis. |
| `enable_buffer_insert_optimization` | The DynamicCV pipeline enables buffer insertion optimization automatically. |
| `enable_cce_vf_auto_sync` | No replacement. |
| `enable_cce_vf_remove_membar` | No replacement. |
| `enable_cross_if_fusion` | No replacement. |
| `enable_drop_unit_dims` | Use `enable_flatten` instead when flattening is intended. |
| `enable_linearize` | No replacement. |
| `enable_simt_reorder_instruction` | Use `simt_optimization_mode`. |
| `enable_vf_fusion` | This switch is no longer forwarded to the NPU compiler. |
| `enable_mask_fallback_conversion` | The backend fixes mask fallback conversion to disabled. |
| `enable_nd2nz_on_vector` | The backend fixes Vector ND2NZ conversion to disabled. |
| `enable_select_analysis` | The backend fixes select analysis to enabled. |
| `enable_sync_block_lock` | No replacement. |
| `enable_ub_refine_opt` | The backend fixes UB refine optimization to disabled. |
| `graph_optimize_emit_remarks` | Graph Optimization remarks are fixed to disabled by the backend. |
| `graph_optimize_max_rewrites_per_function` | The maximum rewrites per function is fixed to 64 by the backend. |
| `graph_optimize_rule_mask` | The old field is ignored. |
| `graph_optimize_ub_capacity_bytes` | The Graph Optimization UB budget is derived from the target product. |
| `grid_num_tiles` | The backend injects this value automatically from a static grid. |
| `has_auto_blockify_blacklist_op` | The compiler derives the safety marker by scanning TTIR. |
| `kernel_name` | The kernel name is derived from TTIR. |
| `llvm_version` | No replacement. |
| `mix_mode` | Derived automatically by the compiler. |
| `ops_reorder` | No replacement. |
| `optimize_dynamic_offset` | The backend fixes dynamic-offset optimization to disabled. |
| `parallel_mode` | Parallel mode is derived from `compile_mode` and Linalg IR. |
| `storage_align` | No replacement. |
| `stream` | Execution streams are managed by the runtime and driver. |
| `use_bytecode` | The bytecode pipeline is always enabled by the backend. |
| `vf_merge_level` | The backend default VF merge level is used. |
| `warp_size` | The Ascend backend fixes the warp size to 32. |

### Deprecated Environment Variables

Remove the following variables from launch scripts or the runtime environment; the compiler ignores their values.

| Deprecated environment variable | Replacement or current behavior |
|---------------------------------|----------------------------------------|
| `LLVM_ROOT` | Set `CC` to select the CPU launcher compiler when needed. |
| `MLIR_ROOT` | Use packaged MLIR tools or tools discoverable through `PATH`. |
| `TRITON_ALL_BLOCKS_PARALLEL` | Automatic block mapping is managed by backend policy. |
| `TRITON_ASCEND_ARCH` | The target architecture is provided by the explicit compilation target's `GPUTarget.arch`. |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | No replacement. |
| `TRITON_BACKEND` | The variable no longer selects Ascend backend policy. |
| `TRITON_DISABLE_FFTS` | The compilation target determines FFTS policy. |
| `TRITON_REGISTER_TENSOR_MSPROF` | Environment variables no longer control tensor-shape registration with msprof. |
