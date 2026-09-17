# 环境变量与编译选项

本文汇总 Triton-Ascend 中可由开发者显式控制的行为开关，包括运行前设置的环境变量，以及编译期通过 `triton.Config` 或 kernel launch meta-parameter 传入的 NPU 编译选项。

## 环境变量

### 环境变量用法示例

环境变量需在运行 Python 程序前设置，例如：

```bash
export TRITON_DEBUG=1
python run_kernel.py
```

### 环境变量参考表

环境变量配置参考下表：

| 类别 | 环境变量 | 默认值 | 功能说明 | 配置说明 | 变更声明 |
|------|----------|--------|----------|----------|----------|
| **调试与日志** | TRITON_DEBUG | 0 或未设置 | 启用 Triton 的调试输出功能，用于在运行时打印详细的调试信息。这对于排查编译或执行阶段的问题非常有用。 当设置为 1 时，Triton 会输出更多关于编译过程、内核生成和执行的信息。 某些实现中可能支持更细粒度的调试级别（如 2, 3 等），具体取决于 Triton 的版本和实现。 | 0：不启用DEBUG<br>1：启用DEBUG | |
| **调试与日志** | MLIR_ENABLE_DUMP | 0 或未设置 | 在每次 MLIR 优化前转储所有内核的 IR。使用 `MLIR_ENABLE_DUMP=kernelName`可以只转储特定内核的IR。 | 0：不转储<br>1：转储所有内核IR kernelName：转储特定内核IR | Triton 缓存可能干扰转储。如果 `MLIR_ENABLE_DUMP=1`  不生效，可尝试清理 Triton 缓存： `rm -r ~/.triton/cache` |
| **调试与日志** | LLVM_IR_ENABLE_DUMP | 0 或未设置 | 在每次 LLVM IR 优化前转储 IR。 | 0：不转储<br>1：转储IR | |
| **调试与日志** | TRITON_REPRODUCER_PATH | 未设置 | 在每个 MLIR 编译阶段前生成 MLIR 复现文件。如果某阶段失败，`<reproducer_path>`  将保存失败前的 MLIR 状态。 | <reproducer_path>：保存路径 | |
| **调试与日志** | TRITON_INTERPRET | 0 或未设置 | 使用 Triton 解释器而非 GPU 运行，支持在核函数代码中插入 Python 断点 | 0：不支持断点<br>1：支持断点 | |
| **调试与日志** | TRITON_ENABLE_LLVM_DEBUG | 0 或未设置 | 向LLVM 传递`-debug`参数，输出大量调试信息。若信息过多，可使用`TRITON_LLVM_DEBUG_ONLY`限制输出范围。 | 0：不传递<br>1：传递 | 另一种减少输出干扰的方法是：先设置 `LLVM_IR_ENABLE_DUMP=1`运行程序，提取目标LLVM优化通道前的中间表示（IR），然后单独运行LLVM的`opt`工具，此时可通过命令行添加`-debug-only=foo`参数来限定调试范围。 |
| **调试与日志** | TRITON_LLVM_DEBUG_ONLY | 未设置 | 功能等同于 LLVM 的`-debug-only`命令行选项。该参数可将 LLVM 调试输出限定到特定的优化通道或组件名称（这些名称通过 LLVM 和 Triton 中的`#define DEBUG_TYPE`宏定义），从而有效减少调试信息的冗余输出。用户可指定一个或多个逗号分隔的值，例如：`TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"`或`TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`。 | 逗号分隔值：通道或组件名称 | |
| **调试与日志** | USE_IR_LOC | 0 或未设置 | 控制是否在生成的中间表示（IR）中包含位置信息（如文件名、行号等）。这些信息对调试很有帮助，但可能会增加生成的IR的大小。设置为1，会重新解析中间表示(IR)，将位置信息映射为具有特定扩展名的IR文件行号（而非Python源文件行号）。这能建立从IR到LLVM IR/PTX的直接映射关系。配合性能分析工具使用时，可实现对IR指令的细粒度性能剖析。 | 0：不包含位置信息<br>1：包含位置信息 | |
| **调试与日志** | TRITON_DISABLE_LINE_INFO | true | 控制是否在`bishengir-compile`命令中追加`--enable-debug-info=true`，即是否在生成的内核可执行文件中包含行号等调试信息。**注意**：triton-ascend 默认值为`true`（默认**关闭**行号信息，与社区 Triton 默认开启相反）。行号信息可用于定位精度/性能问题及 profiling 分析，但会增加编译产物大小。 | 0/false：启用行号信息<br>1/true：禁用行号信息 | 配合性能分析使用时，请参考[算子性能调优方法](./debug_guide/profiling.md)。 |
| **调试与日志** | TRITON_PRINT_AUTOTUNING | 0 或未设置 | 在自动调优完成后，输出每个内核的最佳配置及总耗时。 | 0：不输出<br>1：输出 | |
| **调试与日志** | MLIR_ENABLE_REMARK | 0 或未设置 | 启用MLIR 编译过程中的备注信息输出，包括以备注形式输出的性能警告。 | 0：不启用<br>1：启用 | |
| **调试与日志** | TRITON_KERNEL_DUMP | 0 或未设置 | 启用或禁用 Triton 内核的转储功能，当启用时，Triton 会将生成的内核代码（各编译阶段IR及最终PTX）保存到指定目录。 | 0：不启用<br>1：启用 | |
| **调试与日志** | TRITON_DUMP_DIR | ~/.triton/dump | 指定 Triton 内核转储文件的保存目录。当`TRITON_KERNEL_DUMP=1`时保存IR和PTX的目录。 | "path"：保存路径 | |
| **调试与日志** | TRITON_DEVICE_PRINT | 0 或未设置 | 当设置为`1` 或者 `true`时（`TRUE` 将被转换为 `true`），启用`tl.device_print`功能。 重要说明：该功能使用GM缓冲区（其指针被传递给内核）。 | 0：不启动<br>1：启用`tl.device_print`功能 | 每个线程的GM缓冲区最大为16KB，超限内容将被丢弃。该值目前固定，后续将通过环境变量调整。 |
| **调试与日志** | TRITON_MEMORY_DISPLAY | 0 或未设置 | 控制是否生成内存使用情况的 json 文件。当`TRITON_MEMORY_DISPLAY=1`时保存 memory_info_aic/aiv.json 文件到当前目录 。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_ALWAYS_COMPILE | 0 或未设置 | 控制 Triton 是否每次运行都强制重新编译内核，而不是使用已有的缓存版本。 默认情况下，Triton 会对已经编译过的内核进行缓存（基于参数和配置），以提高性能。 设置为 1 后，Triton 将忽略缓存并每次都重新编译内核，这在调试或测试新编译器特性时非常有用。 | 0：不启用<br>1：每次运行都重新编译所有内核 | |
| **编译控制** | DISABLE_LLVM_OPT | 0 或未设置 | 当设置为 1 时，可以禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化)。当设置为字符串，解析为要禁用的LLVM优化标志列表。例如使用`DISABLE_LLVM_OPT="disable-lsr"`可禁用循环强度优化（该优化在某些存在寄存器压力的内核中可能导致高达10%的性能波动）。 | 0：LLVM 的优化是启用状态<br>1：禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化) <list>:"disable-lsr":禁用循环强度优化 </list>| |
| **编译控制** | MLIR_ENABLE_TIMING | 0 或未设置 | 启用或禁用 MLIR 编译过程中的时间统计功能。 | 0：不启用<br>1：启用 | |
| **编译控制** | LLVM_ENABLE_TIMING | 0 或未设置 | 启用或禁用 LLVM 编译过程中的时间统计功能。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_DEFAULT_FP_FUSION | 1 启用 | 控制是否默认启用浮点运算融合优化，覆盖默认的浮点运算融合行为（如mul+add->fma）。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_KERNEL_OVERRIDE | 0 或未设置 | 启用或禁用 Triton 内核覆盖功能，允许在每个编译阶段开始时用用户指定的外部文件（IR/PTX等）覆盖默认生成的内核代码。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_OVERRIDE_DIR | ~/.triton/override | 指定 Triton 内核覆盖文件的查找目录。当`TRITON_KERNEL_OVERRIDE=1`时加载IR/PTX文件的目录。 | "path"：保存路径 | |
| **编译控制** | TRITON_COMPILE_ONLY | 0 或未设置 | remote_launch时使用，只编译不运行。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_DISABLE_PRECOMPILE | 0 或未设置 | 是否禁用预编译。                                                                                                                                                                                                                                                                                  | 0：启用预编译<br>1：禁用预编译                                                                               | |
| **运行与调度** | TRITON_ENABLE_TASKQUEUE | 1 | 是否开启task_queue。 | 0：不启用<br>1：启用 | |
| **运行与调度** | TRITON_ENABLE_SANITIZER | 0 或未设置 | 是否启用 SANITIZER。 | 0：不启用<br>1：启用 | |
| **运行与调度** | ENABLE_PRINT_UB_BITS | 0 或未设置 | 打开后可以获取当前UB占用量，给inductor使用。 | 0：不启用<br>1：启用 | |
| **运行与调度** | NPU_DEVICE_LIMIT | 未设置 | 用户设置算子运行时最大aicore和vector_core,格式要求：数字以,分割。比如：14,28。| 无默认值| |
| **其他** | TRITON_BENCH_METHOD | 未设置 | 使用昇腾NPU时，将`testing.py`中的`do_bench`切换为`do_bench_npu`（需配合`INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE = 1`使用）。设为`default`时即使NPU可用，仍调用原`do_bench`函数。 | "npu"：切换为`do_bench_npu` | |
| **其他** | TRITON_REMOTE_RUN_CONFIG_PATH | path | 指定远程运行的配置路径。 | 直接给定path | |

## 编译选项

编译选项用于控制单个 Triton kernel 的编译策略，可通过 `triton.Config`、Autotune 参数或 kernel launch meta-parameter 传入。

### 编译选项用法示例

例如，可在 kernel launch 时直接传入 `multibuffer`：

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

### 编译选项参考表

编译选项配置参考下表：

| 类别 | 编译选项 | 默认值/可选值 | 功能说明 | 配置说明 |
|------|----------|----------------|----------|----------|
| **通用流水** | `multibuffer` | `True`（默认）、`False` | 启用或禁用 ping-pong/double buffer 流水。默认开启。 | `triton.Config` 或 launch meta-parameter |
| **图优化** | `enable_graph_optimize` | `True`（默认）、`False` | 启用或禁用 TTIR Graph Optimization。具体规则、重写次数和 UB 预算由后端维护。 | `triton.Config` 或 launch meta-parameter |
| **毕昇编译器** | `bisheng_options` | 后端默认字符串或用户字符串 | 向支持该选项的毕昇编译路径透传附加参数。 | `triton.Config` 或 launch meta-parameter |
| **CV 融合** | `enable_auto_bind_sub_block` | `None`、`True`、`False` | 启用或禁用自动绑定 sub-block。 | `triton.Config` 或 launch meta-parameter |
| **CV 融合** | `enable_hivm_auto_cv_balance` | `None`、`True`、`False` | 启用或禁用自动 CV balance。 | `triton.Config` 或 Autotune 参数 |
| **CV 融合** | `enable_cube_block_merge` | `False`（默认）、`True` | 控制 DynamicCV pipeline 的 Cube block merge。 | `triton.Config` 或 launch meta-parameter |
| **VF 融合** | `vf_fusion_mode` | `None`（默认，遵循 BiShengIR 默认值 `"max-parallel"`）、`"max-parallel"`、`"all-op"`、`"ub-aware-op"` | 选择 Ascend 950 BiShengIR 阶段的 VF fusion 策略。 | `triton.Config` 或 launch meta-parameter |
| **VF 融合** | `enable_vf_fusion` | `None`（默认，遵循 BiShengIR 默认值 `True`）、`True`、`False` | 控制 Ascend 950 BiShengIR 阶段是否启用 VF fusion。该选项是 VF fusion 的总开关；`vf_fusion_mode` 仅用于选择融合策略。 | `triton.Config` 或 launch meta-parameter |
| **HFusion** | `hfusion_enable_multiple_consumer_fusion` | `False`（默认）、`True` | 控制 Ascend 950 BiShengIR 阶段的多 consumer 融合。 | `triton.Config` 或 launch meta-parameter |
| **CV 融合/同步** | `sync_solver` | `None`、`True`、`False` | 启用或禁用 HIVM 同步求解器。 | `triton.Config` 或 launch meta-parameter |
| **同步** | `unit_flag` | `None`、`True`、`False` | Cube 搬出相关同步优化项。 | `triton.Config` 或 Autotune 参数 |
| **同步** | `inject_barrier_all` | `None`、`True`、`False` | 启用或禁用自动注入 barrier 同步。 | `triton.Config` 或 launch meta-parameter |
| **同步** | `inject_block_all` | `None`、`True`、`False` | 启用或禁用自动注入 block 同步。 | `triton.Config` 或 launch meta-parameter |
| **多缓冲范围** | `limit_auto_multi_buffer_only_for_local_buffer` | `None`、`True`、`False` | 限制自动 multi-buffer 只作用于 local buffer。 | `triton.Config` 或 Autotune 参数 |
| **多缓冲范围** | `limit_auto_multi_buffer_of_local_buffer` | `None`、`"no-limit"`、`"no-l0c"` | 配置 local buffer 自动 multi-buffer 的 scope。 | `triton.Config` 或 Autotune 参数 |
| **Workspace** | `set_workspace_multibuffer` | `None`、`2`、`4` | 配置 workspace multi-buffer 档位。 | `triton.Config` 或 Autotune 参数 |
| **CV 融合 tiling** | `tile_mix_vector_loop` | `None`、`2`、`4`、`8` | 配置 Vector loop 的切分份数。 | `triton.Config` 或 Autotune 参数 |
| **CV 融合 tiling** | `tile_mix_cube_loop` | `None`、`2`、`4`、`8` | 配置 Cube loop 的切分份数。 | `triton.Config` 或 Autotune 参数 |
| **DynamicCV 缓冲** | `buf_slot_num_of_veccore` | `None`（默认）或整数 | 配置 veccore 内部 buffer slot 数量。 | `triton.Config` 或 launch meta-parameter |
| **DynamicCV 缓冲** | `buf_slot_num_of_crosscore` | `None`（默认）或整数 | 配置跨 core buffer slot 数量。 | `triton.Config` 或 launch meta-parameter |
| **DynamicCV 缓冲** | `buf_slot_num_of_gm` | `None`（默认）或整数 | 配置 GM load buffer slot 数量。 | `triton.Config` 或 launch meta-parameter |
| **编译模式** | `compile_mode` | `"simd_simt_template"`（默认）、`"simd"`、`"simt_only"` | 控制 SIMD / SIMT 编译路径。`"simd"`：纯 SIMD；`"simd_simt_template"`：普通 SIMD pipeline，并在 Ascend 950 上启用 template-SIMT 子路径；`"simt_only"`：仅 Ascend 950 支持的纯 SIMT 路径（`ttir→npubin`）。 | `triton.Config` 或 launch meta-parameter |

(compiler-option-cleanup-and-compatibility)=

### 编译选项清理与兼容性

环境变量和编译选项参考表仅列出仍然生效的公开控制项。已经删除或转由后端内部维护的旧环境变量和编译选项不再列入参考表。作为临时兼容措施，当前继续传入这些废弃项时，编译器会发出 `FutureWarning` 并忽略用户值，使用后端维护值或默认行为。这些废弃项将在后续版本中按计划彻底删除；废弃编译选项彻底删除后，继续传入将被视为不支持的选项，并导致编译报错。用户应在兼容期内完成迁移。

#### 已更名的编译选项

以下废弃名称仍可在兼容期内传入，编译器会将其路由到当前名称。新代码应直接使用当前名称：

| 废弃名称 | 当前名称或取值 | 兼容行为 |
|----------|----------------|----------|
| `force_simt_only` | `compile_mode="simt_only"` | 显式传入即发出 `FutureWarning`；旧值为 `True` 时路由并覆盖已传入的 `compile_mode`，为 `False` 时不改变当前模式。 |
| `force_simt_template` | `compile_mode="simd_simt_template"` | 显式传入即发出 `FutureWarning`；旧值为 `True` 时路由并覆盖已传入的 `compile_mode`，为 `False` 时不改变当前模式。 |
| `intra_cache_num` | `buf_slot_num_of_veccore` | 发出 `FutureWarning` 后保留原值；同时传入新旧名称时使用新名称。 |
| `inter_cache_num` | `buf_slot_num_of_crosscore` | 发出 `FutureWarning` 后保留原值；同时传入新旧名称时使用新名称。 |
| `load_cache_num` | `buf_slot_num_of_gm` | 发出 `FutureWarning` 后保留原值；同时传入新旧名称时使用新名称。 |

`compile_mode="unstructured_in_simt"` 当前作为 `compile_mode="simd_simt_template"` 的等价兼容取值接受，并会规范化为当前取值；它当前不会触发 `FutureWarning`。新代码应使用 `simd_simt_template`。

#### 应从用户配置中删除的编译选项

以下废弃选项没有继续生效的同名公开入口。兼容期内显式传入时，编译器会发出 `FutureWarning` 并忽略用户值；用户应按下表将其从 `triton.Config`、Autotune 配置或 kernel launch meta-parameter 中删除。

| 废弃选项 | 迁移方式及当前后端行为 |
|----------|--------------------------|
| `add_auto_scheduling` | 删除该选项；DAG 自动调度开关已移除，无替代项。 |
| `allow_fp8e4nv` | 删除该选项；该字段没有有效 consumer，无替代项。 |
| `arch` | 删除该选项；目标架构由编译目标的 `GPUTarget.arch` 提供。 |
| `auto_blockify_size` | 删除该选项；该字段没有有效 consumer，无替代项。 |
| `auto_tile_and_bind_subblock` | 删除该选项；tiling 和 sub-block binding 由 Linalg IR 与 lock 语义推导。 |
| `code_motion` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `compile_on_910_95` | 删除该选项；目标产品由编译目标自动识别。 |
| `disable_auto_inject_block_sync` | 删除该选项；block synchronization injection 由 NPU IR 管理。 |
| `disable_size_align_for_cast` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `enable_auto_blockify` | 删除该选项；自动 block mapping 及安全分析由后端管理。 |
| `enable_buffer_insert_optimization` | 删除该选项；DynamicCV 在内部保持 buffer insertion optimization 开启。 |
| `enable_cce_vf_auto_sync` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `enable_cce_vf_remove_membar` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `enable_cross_if_fusion` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `enable_drop_unit_dims` | 删除该选项；如果目的是启用 flatten，改用 `enable_flatten`。 |
| `enable_linearize` | 删除该选项；该字段没有有效 consumer，无替代项。 |
| `enable_mask_fallback_conversion` | 删除该选项；后端将 mask fallback conversion 固定为关闭。 |
| `enable_nd2nz_on_vector` | 删除该选项；后端将 Vector ND2NZ conversion 固定为关闭。 |
| `enable_select_analysis` | 删除该选项；后端将 select analysis 固定为开启。 |
| `enable_sync_block_lock` | 删除该选项；该字段没有有效 consumer，无替代项。 |
| `enable_ub_refine_opt` | 删除该选项；后端将 UB refine optimization 固定为关闭。 |
| `graph_optimize_emit_remarks` | 删除该选项；Graph Optimization remarks 由后端固定为关闭。 |
| `graph_optimize_max_rewrites_per_function` | 删除该选项；每个函数的最大重写次数由后端固定为 64。 |
| `graph_optimize_rule_mask` | 删除该选项；Graph Optimization rule mask 由后端固定为 511。 |
| `graph_optimize_ub_capacity_bytes` | 删除该选项；Graph Optimization UB 预算由后端根据目标产品推导。 |
| `grid_num_tiles` | 删除该选项；该值由后端根据静态 grid 自动注入。 |
| `has_auto_blockify_blacklist_op` | 删除该选项；安全标记由编译器扫描 TTIR 生成。 |
| `kernel_name` | 删除该选项；kernel 名称由 TTIR 推导。 |
| `llvm_version` | 删除该选项；该字段没有有效 consumer，无替代项。 |
| `mix_mode` | 删除该选项；mix mode 由 Linalg IR 推导为内部 metadata。 |
| `ops_reorder` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `optimize_dynamic_offset` | 删除该选项；后端将 dynamic-offset optimization 固定为关闭。 |
| `parallel_mode` | 删除该选项；parallel mode 由 `compile_mode` 和 Linalg IR 推导。 |
| `storage_align` | 删除该选项；原 vendor compiler 控制项已移除，无替代项。 |
| `stream` | 删除该选项；launch stream 由 runtime 和 driver 管理。 |
| `use_bytecode` | 删除该选项；bytecode pipeline 由后端固定启用。 |
| `vf_merge_level` | 删除该选项；VF merge level 使用后端默认值。 |
| `warp_size` | 删除该选项；Ascend 后端将 warp size 固定为 32。 |

#### 已废弃的环境变量

以下环境变量在兼容期内被检测到时会发出 `FutureWarning`，但其值不再影响后端行为。用户应取消设置：

| 废弃环境变量 | 迁移方式及当前后端行为 |
|----------------|--------------------------|
| `LLVM_ROOT` | 取消设置；如需选择 CPU launcher 编译器，设置 `CC`。 |
| `MLIR_ROOT` | 取消设置；使用随包提供或可从 `PATH` 发现的 MLIR 工具。 |
| `TRITON_ALL_BLOCKS_PARALLEL` | 取消设置；自动 block mapping 由后端策略管理。 |
| `TRITON_ASCEND_ARCH` | 取消设置；目标架构由显式编译目标的 `GPUTarget.arch` 提供。 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | 取消设置；该变量没有有效 consumer，无替代项。 |
| `TRITON_BACKEND` | 取消设置；Ascend backend policy 不再由该环境变量选择。 |
| `TRITON_DISABLE_FFTS` | 取消设置；FFTS policy 根据显式编译目标推导。 |
| `TRITON_REGISTER_TENSOR_MSPROF` | 取消设置；tensor-shape msprof registration 不再由环境变量控制。 |
