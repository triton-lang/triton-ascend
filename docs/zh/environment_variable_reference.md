# 环境变量

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
| **调试与日志** | TRITON_PRINT_AUTOTUNING | 0 或未设置 | 在自动调优完成后，输出每个内核的最佳配置及总耗时。 | 0：不输出<br>1：输出 | |
| **调试与日志** | MLIR_ENABLE_REMARK | 0 或未设置 | 启用MLIR 编译过程中的备注信息输出，包括以备注形式输出的性能警告。 | 0：不启用<br>1：启用 | |
| **调试与日志** | TRITON_KERNEL_DUMP | 0 或未设置 | 启用或禁用 Triton 内核的转储功能，当启用时，Triton 会将生成的内核代码（各编译阶段IR及最终PTX）保存到指定目录。 | 0：不启用<br>1：启用 | |
| **调试与日志** | TRITON_DUMP_DIR | 当前工作目录或未设置 | 指定 Triton 内核转储文件的保存目录。当`TRITON_KERNEL_DUMP=1`时保存IR和PTX的目录。 | "path"：保存路径 | |
| **调试与日志** | TRITON_DEVICE_PRINT | 0 或未设置 | 当设置为`1` 或者 `true`时（`TRUE` 将被转换为 `true`），启用`tl.device_print`功能。 重要说明：该功能使用GM缓冲区（其指针被传递给内核）。 | 0：不启动<br>1：启用`tl.device_print`功能 | 每个线程的GM缓冲区最大为16KB，超限内容将被丢弃。该值目前固定，后续将通过环境变量调整。 |
| **调试与日志** | TRITON_MEMORY_DISPLAY | 0 或未设置 | 控制是否生成内存使用情况的 json 文件。当`TRITON_MEMORY_DISPLAY=1`时保存 memory_info_aic/aiv.json 文件到当前目录 。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_ALWAYS_COMPILE | 0 或未设置 | 控制 Triton 是否每次运行都强制重新编译内核，而不是使用已有的缓存版本。 默认情况下，Triton 会对已经编译过的内核进行缓存（基于参数和配置），以提高性能。 设置为 1 后，Triton 将忽略缓存并每次都重新编译内核，这在调试或测试新编译器特性时非常有用。 | 0：不启用<br>1：每次运行都重新编译所有内核 | |
| **编译控制** | DISABLE_LLVM_OPT | 0 或未设置 | 当设置为 1 时，可以禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化)。当设置为字符串，解析为要禁用的LLVM优化标志列表。例如使用`DISABLE_LLVM_OPT="disable-lsr"`可禁用循环强度优化（该优化在某些存在寄存器压力的内核中可能导致高达10%的性能波动）。 | 0：LLVM 的优化是启用状态<br>1：禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化) <list>:"disable-lsr":禁用循环强度优化 </list>| |
| **编译控制** | MLIR_ENABLE_TIMING | 0 或未设置 | 启用或禁用 MLIR 编译过程中的时间统计功能。 | 0：不启用<br>1：启用 | |
| **编译控制** | LLVM_ENABLE_TIMING | 0 或未设置 | 启用或禁用 LLVM 编译过程中的时间统计功能。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_DEFAULT_FP_FUSION | 1 启用 | 控制是否默认启用浮点运算融合优化，覆盖默认的浮点运算融合行为（如mul+add->fma）。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_KERNEL_OVERRIDE | 0 或未设置 | 启用或禁用 Triton 内核覆盖功能，允许在每个编译阶段开始时用用户指定的外部文件（IR/PTX等）覆盖默认生成的内核代码。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_OVERRIDE_DIR | 当前工作目录或未设置 | 指定 Triton 内核覆盖文件的查找目录。当`TRITON_KERNEL_OVERRIDE=1`时加载IR/PTX文件的目录。 | "path"：保存路径 | |
| **编译控制** | TRITON_ASCEND_COMPILE_SPEED_OPT | 0 或未设置 | 控制JIT编译器在发现内核编译失败后是否跳过后续编译阶段。设为`1`跳过（默认`0`继续尝试）。 | 0：继续尝试<br>1：跳过 | |
| **编译控制** | TRITON_COMPILE_ONLY | 0 或未设置 | remote_launch时使用，只编译不运行。 | 0：不启用<br>1：启用 | |
| **编译控制** | TRITON_DISABLE_FFTS | 0 或未设置 | 是否禁用FFTS。 | 0：启用<br>1：禁用 | |
| **编译控制** | TRITON_DISABLE_PRECOMPILE | 0 或未设置 | 是否禁用预编译。                                                                                                                                                                                                                                                                                  | 0：使能预编译<br>1：禁用预编译                                                                               | |
| **运行与调度** | TRITON_ALL_BLOCKS_PARALLEL | 0 或未设置 | 启用或禁用自动根据物理核数优化逻辑核数，仅当逻辑核间可并行时方可启动。当逻辑核数大于物理核数时，启动该优化，则编译器自动调整逻辑核数量为物理核数，减少调度开销；使能后允许grid>65535。限制：triton kernel的逻辑必须对执行顺序不敏感才能开启该选项，否则可能会导致死锁。 | 0：不启用<br>1：启用 | |
| **运行与调度** | TRITON_ENABLE_TASKQUEUE | 0 或未设置 | 是否开启task_queue。 | 0：不启用<br>1：启用 | |
| **运行与调度** | TRITON_ENABLE_SANITIZER | 0 或未设置 | 是否使能SANITIZER。 | 0：不启用<br>1：启用 | |
| **运行与调度** | ENABLE_PRINT_UB_BITS | 0 或未设置 | 打开后可以获取当前UB占用量，给inductor使用。 | 0：不启用<br>1：启用 | |
| **其他** | TRITON_BENCH_METHOD | 未设置 | 使用昇腾NPU时，将`testing.py`中的`do_bench`切换为`do_bench_npu`（需配合`INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE = 1`使用）。设为`default`时即使NPU可用，仍调用原`do_bench`函数。 | "npu"：切换为`do_bench_npu` | |
| **其他** | TRITON_REMOTE_RUN_CONFIG_PATH | path | 指定远程运行的配置路径。 | 直接给定path | |
