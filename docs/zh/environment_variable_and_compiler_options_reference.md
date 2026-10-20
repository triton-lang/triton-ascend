# 环境变量与编译选项

本文面向Triton-Ascend算子开发者，提供编译、运行、调试和自动调优配置的查阅入口，说明各配置项的用途、默认值及A2/A3、950支持情况。环境变量用于设置进程内的编译与运行行为，编译选项用于配置单个kernel。旧配置的迁移方式集中在文末的[编译选项清理与兼容性](#compiler-option-cleanup-and-compatibility)。

## 环境变量

### 环境变量用法示例

在启动Python程序前，使用 `export` 设置环境变量。例如，设置 `TRITON_DEBUG=1` 开启调试功能：

```bash
export TRITON_DEBUG=1
python3 run_kernel.py
```

`export` 设置的变量会被当前终端后续启动的程序继承。若只需为一次运行设置变量，可将赋值写在命令前：

```bash
TRITON_DEBUG=1 python3 run_kernel.py
```

### 环境变量参考表

950列包含Ascend 950PR/950DT；“—”表示不适用，编译模式限制见[编译选项参考表](#compiler-options-reference)。布尔环境变量除特别注明外可用 `1` 启用。

| 类别 | 环境变量 | 功能说明 | 默认值 | A2/A3 | 950 |
|------|----------|----------|--------|-------|-----|
| 调试 | `TRITON_DEBUG` | 开启JIT调试及设备断言，保存TTIR；常规路径还输出编译日志并保存后端IR与二进制。 | 关闭 | 支持 | 支持 |
| 调试 | `USE_IR_LOC` | 以指定阶段的IR文件行号重建位置信息；当前AST编译可用 `ttir`，IR输入还可使用源文件位置快照。 | 未设置 | 支持 | 支持 |
| 调试 | `TRITON_FRONT_END_DEBUGGING` | 前端错误发生时保留原始Python异常堆栈。 | 关闭 | 支持 | 支持 |
| 调试 | `TRITON_KERNEL_DUMP` | 保存编译阶段产物及最终 `npubin`；只对实际发生的编译生效。 | 关闭 | 支持 | 支持 |
| 调试 | `TRITON_DUMP_DIR` | 指定 `TRITON_KERNEL_DUMP` 的输出目录。 | `$TRITON_HOME/.triton/dump` | 支持 | 支持 |
| 调试 | `TRITON_DISABLE_LINE_INFO` | `true/1` 禁止向毕昇编译器请求行号信息；`false/0` 启用。 | `true` | 支持 | 常规路径 |
| 调试 | `LLVM_EXTRACT_DI_LOCAL_VARIABLES` | 请求毕昇编译器生成调试变量信息。 | 关闭 | 支持 | 常规路径 |
| 调试 | `TRITON_DEVICE_PRINT` | 启用 `tl.device_print` 的设备端GM打印缓冲区。 | 关闭 | 支持 | 支持 |
| 调试 | `TRITON_MEMORY_DISPLAY` | 请求编译器输出内存使用信息。 | 关闭 | 支持 | — |
| 调试 | `ENABLE_PRINT_UB_BITS` | 请求编译器打印UB占用位数。 | 关闭 | 支持 | 常规路径 |
| 调试 | `TRITON_INTERPRET` | 在CPU解释器中执行其支持的Triton操作，便于Python断点调试。 | 关闭 | 支持 | 支持 |
| 调试 | `TRITON_ENABLE_SANITIZER` | 向Ascend NPU编译器传入sanitizer开关。 | 关闭 | 支持 | 常规路径 |
| 编译 | `TRITON_ALWAYS_COMPILE` | 忽略命中的编译缓存，强制重新编译。 | 关闭 | 支持 | 支持 |
| 编译 | `TRITON_KERNEL_OVERRIDE` | 从覆盖目录读取与编译阶段同名的外部IR/二进制文件。 | 关闭 | 支持 | 支持 |
| 编译 | `TRITON_OVERRIDE_DIR` | 指定覆盖文件目录。 | `$TRITON_HOME/.triton/override` | 支持 | 支持 |
| 编译 | `TRITON_F32_DEFAULT` | 设置 `tl.dot` 的默认F32输入精度；Ascend使用 `ieee`、`hf32`，兼容输入 `tf32` 会映射为 `hf32`。 | 未设置，使用后端的 `ieee` | 支持 | 支持 |
| 编译 | `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | 允许JIT前端引用非 `constexpr` 全局变量。 | 关闭 | 支持 | 支持 |
| 编译 | `TRITON_ENABLE_LIBDEVICE` | 在A2/A3编译路径额外链接libdevice；非空字符串均视为开启，包括 `"0"`。 | 未设置 | 支持 | — |
| 编译 | `TRITON_ENABLE_LIBDEVICE_SIMT` | 在950上为libdevice调用选择SIMT版本；非空字符串均视为开启，包括 `"0"`。 | 未设置 | — | 支持 |
| 缓存与调优 | `TRITON_HOME` | 设置Triton缓存、dump和override目录的根位置。 | 用户主目录 | 支持 | 支持 |
| 缓存与调优 | `TRITON_CACHE_DIR` | 指定编译缓存目录。 | `$TRITON_HOME/.triton/cache` | 支持 | 支持 |
| 缓存与调优 | `TRITON_CACHE_AUTOTUNING` | 缓存autotune的选择结果。 | 关闭 | 支持 | 支持 |
| 缓存与调优 | `TRITON_PRINT_AUTOTUNING` | 设为 `1` 时打印autotune结果及常规路径的编译命令。 | 关闭 | 支持 | 支持 |
| 缓存与调优 | `TRITON_CACHE_MANAGER` | 用 `MODULE:CLASS` 指定编译缓存管理器；远程缓存设为 `triton.runtime.cache:RemoteCacheManager`。 | 本地缓存管理器 | 支持 | 支持 |
| 缓存与调优 | `TRITON_REMOTE_CACHE_BACKEND` | 用 `MODULE:CLASS` 指定远程缓存的存储后端，需配合远程缓存管理器使用。 | 未设置 | 支持 | 支持 |
| 缓存与调优 | `TRITON_ENABLE_UBTUNER` | 选择UB调优模式：`compile` 或 `run`。 | 未设置 | 支持 | 常规路径 |
| 缓存与调优 | `TRITON_AUTOTUNE_PARALLEL_COMPILE` | 控制Ascend autotune的并行编译，`0` 关闭。 | `1` | 支持 | 支持 |
| 缓存与调优 | `TRITON_BENCH_METHOD` | 设为 `npu` 时选择Ascend的 `do_bench_npu`；`default` 使用通用测量方法。 | `default` | 支持 | 支持 |
| 运行 | `TRITON_COMPILE_ONLY` | 设为 `1/true` 时不启动kernel；编译及launcher扩展加载仍会执行。 | 关闭 | 支持 | 支持 |
| 运行 | `TRITON_ENABLE_TASKQUEUE` | 控制Ascend launcher的task queue。 | 开启 | 支持 | 支持 |
| 运行 | `TRITON_GRID_WARN_PRINT` | 打印Ascend launcher的grid大小警告。 | 关闭 | 支持 | 支持 |
| 运行 | `NPU_DEVICE_LIMIT` | 按 `AI Core数,Vector Core数` 限制可用核心数，例如 `14,28`；需符合设备比例。 | 未设置，使用设备值 | 支持 | 支持 |
| 工具链选择 | `CC` | 设置CPU launcher扩展的C/C++编译器。 | 未设置，使用系统默认编译器 | 支持 | 支持 |
| 工具链选择 | `TRITON_NPU_COMPILER_PATH` | 仅当随包工具与 `PATH` 查找均失败时，作为Ascend编译器工具目录的后备路径。 | 未设置 | 支持 | 支持 |
| 工具链选择 | `TRITON_BUILD_DIR` | 查找 `triton-opt`、`triton-mlir-opt` 时，在随包工具之后、`PATH` 之前查找该目录的 `bin/`。 | 未设置 | 支持 | 支持 |

## 编译选项

### 编译选项用法示例

编译选项可以直接传入 `kernel[grid](...)`，也可以通过 `triton.Config` 设置自动调优的候选配置。`triton.Config` 的 `num_warps`、`num_stages` 和 `ir_override` 使用独立构造参数，其余受支持的编译选项放在 `kwargs` 中。

示例中，`BLOCK_SIZE` 是kernel自定义的 `tl.constexpr` 参数，`compile_mode` 和 `num_stages` 是编译选项：

```python
import triton

kernel[grid](*args, BLOCK_SIZE=1024, compile_mode="simd", num_stages=1)

configs = [
    triton.Config({"BLOCK_SIZE": 1024, "compile_mode": "simd"}, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024, "compile_mode": "simd"}, num_stages=2),
]
```

候选配置的使用方法见 [Autotune使用指南](./autotune_guide.md)。

<a id="compiler-options-reference"></a>

### 编译选项参考表

表中默认值以Ascend后端为准，`triton.Config` 的差异单独注明；`None` 通常表示由后端或工具链决定。

- **常规路径**：`compile_mode="simd"` 或 `"simd_simt_template"`。
- **仅纯SIMT**：`compile_mode="simt_only"`，仅950支持。
- **需DynamicCV生效**：编译器成功应用了 [DynamicCV流水优化](./architecture_design_and_core_features.md#dynamic-cv-pipeline)，即950常规路径中启用 `enable_dynamic_cv_pipeline=True`，且未发生回退。

| 类别 | 编译选项 | 功能说明 | 默认值/可选值 | A2/A3 | 950 |
|------|----------|----------|---------------|-------|-----|
| Triton通用 | `debug` | 开启设备断言与编译调试；JIT下与环境变量 `TRITON_DEBUG` 任一开启即生效。 | `False`；布尔值 | 支持 | 支持 |
| Triton通用 | `sanitize_overflow` | 启用整数溢出检查；检查通过设备断言执行，还需启用 `debug`。 | `True`；布尔值 | 支持 | 支持 |
| Triton通用 | `num_stages` | 设为 `1` 时关闭自动多缓冲；其他值使用自动模式，不指定各层缓冲数量。 | 未设置 `multibuffer_mode` 时：后端 `2`，`triton.Config` 为 `3`；整数 | 支持 | 常规路径 |
| Triton通用 | `num_warps` | 设置纯SIMT编译器的warp数。 | 后端 `32`；`triton.Config` 为 `4`；正整数 | — | 仅纯SIMT |
| Triton通用 | `ir_override` | 用给定文件覆盖同扩展名的编译阶段产物。 | `None`；文件路径 | 支持 | 支持 |
| 编译模式与图优化 | `compile_mode` | `simd` 使用SIMD；`simd_simt_template` 在A2/A3使用SIMD，在950启用模板SIMT子路径；`simt_only` 使用纯SIMT。 | `"simd_simt_template"`；另可选 `"simd"`、`"simt_only"` | 支持前两项 | 支持全部 |
| 编译模式与图优化 | `enable_graph_optimize` | 控制TTIR Graph Optimization。 | `True`；布尔值 | 支持 | 支持 |
| 多缓冲 | `multibuffer` | 控制自动多缓冲总开关；`num_stages=1` 也会将其关闭。 | `True`；布尔值 | 支持 | 常规路径 |
| 多缓冲 | `multibuffer_mode` | 按 `gm`、`l1`、`l0c`、`ub` 设置多缓冲槽数；`1` 关闭该层多缓冲，大于 `1` 指定槽数。受 `multibuffer` 总开关控制，不能与 `num_stages` 同时显式设置。 | `None`；必须包含四个层级的字典，值为正的32位有符号整数 | 支持 | 常规路径 |
| 多缓冲 | `set_workspace_multibuffer` | 设置workspace多缓冲级别；显式设置 `multibuffer_mode` 且 `gm>1` 时，NPU IR使用该GM槽数覆盖本值。 | `None`；整数，具体范围由NPU编译器决定 | 支持 | 常规路径 |
| 同步 | `sync_solver` | 控制HIVM graph sync solver。 | `None`；布尔值 | 支持 | 常规路径 |
| 同步 | `unit_flag` | 控制HIVM unit-flag同步。 | `None`；布尔值 | 支持 | 常规路径 |
| 同步 | `inject_barrier_all` | 控制自动注入barrier同步。 | `None`；布尔值 | 支持 | 常规路径 |
| 同步 | `inject_block_all` | 控制自动注入block同步。 | `None`；布尔值 | 支持 | 常规路径 |
| 同步 | `disable_auto_inject_block_sync` | 禁用编译器自动插入核间同步。使用Ascend亲和接口 [`sync_block_all`](./triton_api_extension/al/sync_block_all.md)、[`sync_block_set`](./triton_api_extension/al/sync_block_set.md) 或 [`sync_block_wait`](./triton_api_extension/al/sync_block_wait.md) 时，必须设置 `disable_auto_inject_block_sync=True`。 | `None`；布尔值 | 支持 | 常规路径 |
| 调度与流水 | `enable_auto_bind_sub_block` | 覆盖后端推导的sub-block自动绑定策略。 | `None`；布尔值 | 支持 | 常规路径 |
| 调度与流水 | `enable_hivm_auto_cv_balance` | 控制HIVM的自动CV balance。 | `None`；布尔值 | 支持 | 常规路径 |
| 调度与流水 | `tile_mix_vector_loop` | 设置Vector循环的切分份数。 | `None`；整数 | 支持 | — |
| 调度与流水 | `tile_mix_cube_loop` | 设置Cube循环的切分份数。 | `None`；整数 | 支持 | — |
| 调度与流水 | `enable_mixed_cv` | 控制NPU编译器的mixed CV路径。 | `None`；布尔值 | — | 常规路径 |
| 调度与流水 | `enable_dynamic_cv_pipeline` | 启用950的 [DynamicCV流水优化](./architecture_design_and_core_features.md#dynamic-cv-pipeline)。优化生效时设置 `set_workspace_multibuffer=0`、`enable_mixed_cv=True`、`disable_auto_inject_block_sync=True`；若无法应用该优化，则恢复这些选项的原值并继续编译。 | `None`，按目标解析：A2/A3 `False`、950 `True`；布尔值 | — | 常规路径 |
| 调度与流水 | `buf_slot_num_of_veccore` | 设置Vector Core本地缓冲槽数量。 | `None`；整数 | — | 需DynamicCV生效 |
| 调度与流水 | `buf_slot_num_of_crosscore` | 设置跨核缓冲槽数量。 | `None`；整数 | — | 需DynamicCV生效 |
| 调度与流水 | `buf_slot_num_of_gm` | 设置GM加载缓冲槽数量。 | `None`；整数 | — | 需DynamicCV生效 |
| 内存优化 | `enable_ubuf_saving` | 控制UB saving。 | `None`；布尔值 | 支持 | — |
| 内存优化 | `disable_size_align_for_cast` | 控制cast的size-align优化。 | `None`；布尔值 | 支持 | — |
| 内存优化 | `disable_tightly_coupled_buffer_reuse` | 禁止NPU编译器复用tightly coupled buffer。 | `False`；布尔值 | — | 常规路径 |
| 内存优化 | `plan_memory_strategy` | 选择内存规划策略。 | `None`；`"default"`、`"largest-first"` | — | 常规路径 |
| 算子优化 | `enable_preload` | 控制NPU编译器的preload优化。 | `None`；布尔值 | 支持 | 常规路径 |
| 算子优化 | `enable_flatten` | 控制NPU编译器的flatten pass。 | `None`；布尔值 | 支持 | 常规路径 |
| 算子优化 | `enable_auto_vectorize_v2` | 控制NPU编译器的AutoVectorizeV2。 | `None`；布尔值 | 支持 | 常规路径 |
| 算子优化 | `vf_fusion_mode` | 选择VF fusion策略。 | `None`，不传开关；`"max-parallel"`、`"all-op"`、`"ub-aware-op"` | — | 常规路径 |
| 算子优化 | `enable_vf_stack_limit` | 在BiShengIR编译中根据向量函数（VF）的栈占用估算限制融合，降低栈溢出风险。 | `False`；布尔值 | — | 常规路径 |
| 算子优化 | `hfusion_enable_multiple_consumer_fusion` | 控制多个消费者的HFusion。 | `None`；布尔值 | — | 常规路径 |
| 算子优化 | `enable_hivm_batch_matmul` | 为BiShengIR编译启用batch matmul。 | `False`；布尔值 | — | 常规路径 |
| 算子优化 | `auto_vectorize_v2_max_fused_ops_num` | 设置AutoVectorizeV2最大融合op数。 | `None`；整数 | — | 常规路径 |
| 算子优化 | `prevec_max_fused_ops_num` | 设置预向量化最大融合op数。 | `None`；整数 | — | 常规路径 |
| 纯SIMT | `simt_optimization_mode` | 设置纯SIMT的多位优化模式；在A2/A3或950常规路径显式传入时，会发出 `UserWarning` 并忽略。 | `0`；整数，位语义由NPU编译器定义 | — | 仅纯SIMT |
| 纯SIMT | `shared_mem_dynamic_size` | 设置纯SIMT动态共享内存大小。 | `122880` 字节；可显式设整数字节数 | — | 仅纯SIMT |
| 纯SIMT | `disable_fma` | 在纯SIMT编译路径禁用FMA优化。 | `False`；布尔值 | — | 仅纯SIMT |
| 纯SIMT | `superblock_factor` | 在自动block mapping生效且未做row coalescing时，设置纯SIMT superblock因子。 | `1`；大于 `1` 时传给编译器 | — | 仅纯SIMT |

<a id="compiler-option-cleanup-and-compatibility"></a>

## 编译选项清理与兼容性

本节列出的旧编译选项仅在兼容期内接受。**未来版本彻底移除这些选项后，继续通过kernel启动参数或 `triton.Config` 传入它们将导致报错。** 请在升级前按下表迁移或删除旧选项。

由Triton-Ascend处理的更名和废弃项会发出 `FutureWarning`，同一名称在进程内只告警一次。旧多缓冲选项由NPU IR输出废弃提示。

### 旧多缓冲选项迁移

以下三个选项由NPU IR标记为废弃。兼容期内仍按旧语义生效；同时设置 `multibuffer_mode` 时，旧选项的取值被忽略。可按编译器提示的等效配置迁移，并在字典中完整指定四个层级：

| 旧选项 | `multibuffer_mode` 中的对应设置 |
|--------|--------------------------------|
| `limit_auto_multi_buffer_only_for_local_buffer` | `True` 对应 `gm=1`，关闭GM多缓冲。 |
| `limit_auto_multi_buffer_of_local_buffer` | `"no-l0c"` 对应 `l0c=1`；`"no-limit"` 对应 `l0c>1`。 |
| `limit_auto_multi_buffer_buffer` | 950常规路径：`"only-cube"` 对应 `ub=1`；`"only-vector"` 对应 `l1=1`、`l0c=1`；`"no-limit"` 对应 `l1>1`、`ub>1`。 |

例如，仅为L1配置双缓冲：

```python
triton.Config({"multibuffer_mode": {"gm": 1, "l1": 2, "l0c": 1, "ub": 1}})
```

### 已更名的编译选项

将以下旧名称替换为对应的名称或取值：

| 废弃名称 | 当前名称或取值 | 兼容行为 |
|----------|----------------|----------|
| `force_simt_only` | `compile_mode="simt_only"` | 旧值为 `True` 时覆盖已传入的 `compile_mode`，为 `False` 时不改变模式。两个旧force开关都为 `True` 时，本项优先。 |
| `force_simt_template` | `compile_mode="simd_simt_template"` | 旧值为 `True` 时覆盖已传入的 `compile_mode`，为 `False` 时不改变模式。 |
| `intra_cache_num` | `buf_slot_num_of_veccore` | 沿用原值；同时传入新旧名称时使用新名称。 |
| `inter_cache_num` | `buf_slot_num_of_crosscore` | 沿用原值；同时传入新旧名称时使用新名称。 |
| `load_cache_num` | `buf_slot_num_of_gm` | 沿用原值；同时传入新旧名称时使用新名称。 |
| `enable_bishengir_simt_optimization` | `simt_optimization_mode` | 沿用原值；同时传入新旧名称时使用新名称。仅在950纯SIMT路径生效。 |

兼容取值 `compile_mode="unstructured_in_simt"` 等同于 `compile_mode="simd_simt_template"`，仍可使用且不触发 `FutureWarning`。新代码使用 `simd_simt_template`。

### 已废弃的编译选项

从kernel启动参数和 `triton.Config` 中删除以下选项；编译器会忽略其取值。

| 废弃选项 | 替代方式或当前行为 |
|----------|--------------------------|
| `add_auto_scheduling` | DAG自动调度开关已移除，无替代项。 |
| `allow_fp8e4nv` | 无替代项。 |
| `arch` | 目标架构由编译目标的 `GPUTarget.arch` 提供。 |
| `auto_blockify_size` | 无替代项。 |
| `auto_tile_and_bind_subblock` | 切分与sub-block绑定由编译器自动决定。 |
| `bisheng_options` | 用户值被忽略。 |
| `code_motion` | 无替代项。 |
| `compile_on_910_95` | 目标产品由编译目标自动识别。 |
| `enable_auto_blockify` | 自动block映射及安全分析由编译器管理。 |
| `enable_buffer_insert_optimization` | DynamicCV流水自动启用缓冲插入优化。 |
| `enable_cce_vf_auto_sync` | 无替代项。 |
| `enable_cce_vf_remove_membar` | 无替代项。 |
| `enable_cross_if_fusion` | 无替代项。 |
| `enable_drop_unit_dims` | 如果目的是启用flatten，改用 `enable_flatten`。 |
| `enable_linearize` | 无替代项。 |
| `enable_simt_reorder_instruction` | 改用 `simt_optimization_mode`。 |
| `enable_vf_fusion` | 不再向NPU编译器透传。 |
| `enable_mask_fallback_conversion` | 后端将mask fallback conversion固定为关闭。 |
| `enable_nd2nz_on_vector` | 后端将Vector ND2NZ conversion固定为关闭。 |
| `enable_select_analysis` | 后端将select analysis固定为开启。 |
| `enable_sync_block_lock` | 无替代项。 |
| `enable_ub_refine_opt` | 后端将UB refine optimization固定为关闭。 |
| `graph_optimize_emit_remarks` | Graph Optimization remarks由后端固定为关闭。 |
| `graph_optimize_max_rewrites_per_function` | 每个函数的最大重写次数由后端固定为64。 |
| `graph_optimize_rule_mask` | 旧字段的取值被忽略。 |
| `graph_optimize_ub_capacity_bytes` | Graph Optimization UB预算由后端根据目标产品推导。 |
| `grid_num_tiles` | 该值由后端根据静态grid自动注入。 |
| `has_auto_blockify_blacklist_op` | 安全标记由编译器扫描TTIR生成。 |
| `kernel_name` | kernel名称由TTIR推导。 |
| `llvm_version` | 无替代项。 |
| `mix_mode` | 由编译器自动推导。 |
| `ops_reorder` | 无替代项。 |
| `optimize_dynamic_offset` | 后端将dynamic-offset optimization固定为关闭。 |
| `parallel_mode` | parallel mode由 `compile_mode` 和Linalg IR推导。 |
| `storage_align` | 无替代项。 |
| `stream` | 执行流由运行时和驱动管理。 |
| `use_bytecode` | bytecode pipeline由后端固定启用。 |
| `vf_merge_level` | VF merge level使用后端默认值。 |
| `warp_size` | Ascend后端将warp size固定为32。 |

### 已废弃的环境变量

从启动脚本或运行环境中移除以下变量；编译器会忽略其取值。

| 废弃环境变量 | 替代方式或当前行为 |
|----------------|--------------------------|
| `LLVM_ROOT` | 如需选择CPU launcher编译器，设置 `CC`。 |
| `MLIR_ROOT` | 使用随包提供或可从 `PATH` 发现的MLIR工具。 |
| `TRITON_ALL_BLOCKS_PARALLEL` | 自动block mapping由后端策略管理。 |
| `TRITON_ASCEND_ARCH` | 目标架构由显式编译目标的 `GPUTarget.arch` 提供。 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | 无替代项。 |
| `TRITON_BACKEND` | 该变量不再用于选择Ascend后端策略。 |
| `TRITON_DISABLE_FFTS` | FFTS策略由编译目标决定。 |
| `TRITON_REGISTER_TENSOR_MSPROF` | 不再通过环境变量控制msprof的张量形状注册。 |
