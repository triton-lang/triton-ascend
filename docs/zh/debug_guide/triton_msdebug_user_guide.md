# Triton msDebug 使用指南

## 简介

MindStudio Debugger（msDebug）用于调试运行在昇腾 NPU 上的设备程序，支持源码断点、单步、变量与内存查看、寄存器读取、核切换、运行状态查询和异常算子 Core Dump 解析等功能。

msDebug 适合排查以下问题：

- Triton Kernel 运行结果错误；
- `tl.load`、计算或 `tl.store` 阶段的数据不符合预期；
- 多 Block、多核场景中，部分核的计算结果异常；
- Kernel 运行卡住，需要确认停止位置；
- NPU 发生 MTE、VEC、CUBE 等硬件异常；
- 需要分析异常现场生成的 `.core` 文件；
- 源码断点、行号映射或变量调试信息异常。

msDebug 不用于调试 Triton 编译器 Pass 本身。如果程序在生成 `kernel.o` 之前已经编译失败，应使用 IR Dump、编译日志以及 GDB/LLDB 调试 `triton-opt` 或编译器进程。

本文基于以下官方资料整理：

- [MindStudio Debugger 工具用户指南](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)
- [MindStudio 8.3：msDebug 工具介绍](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0062.html?framework=mindspore)

## 使用前准备

### 环境准备

- 请参考 [MindStudio Debugger 安装指南](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/install_guide/msdebug_install_guide.md)安装 msDebug 工具。
- 若要使能 msDebug 工具，需通过以下两种方法安装 NPU 驱动固件（CANN 8.1.RC1 之后的版本且驱动为 25.0.RC1 之后的版本，推荐使用方法一）。

**方法一**：驱动安装时指定 `--full` 参数，然后再使用 root 用户执行 `echo 1 > /proc/debug_switch` 命令启用调试通道，msDebug 工具便可正常使用。

```bash
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full
```

**方法二**：驱动安装时指定 `--debug` 参数，具体安装操作请参见《CANN 软件安装指南》中的“[安装 NPU 驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)”章节。

```bash
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --debug
```

### 为 Triton Kernel 生成调试信息

当前 Triton-Ascend 启用 msDebug 功能需要开启以下环境变量：

```bash
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1
export TRITON_DISABLE_LINE_INFO=0
```

| 环境变量 | 用途 |
|---|---|
| `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` | 开启 Ascend 后端局部变量调试流程，并启用相关 Location、NOP 和行表处理 |
| `TRITON_DISABLE_LINE_INFO=0` | 开启源码行号调试信息；当前后端默认可能关闭该信息，因此需要显式设置 |

当前实现中，Ascend 后端会据此向后端编译器传递 `--enable-debug-info=true` 和 `--enable-debug-variables=true`。

### 约束

- 调试通道权限较高，不建议在生产环境长期启用；
- 同一 Device 建议只启动一个 msDebug 会话，并暂停其他算子任务；
- 被调 Python 程序最好只调用一个目标 Triton Kernel；
- 首次排查时建议关闭 Autotune，固定一组 `BLOCK_SIZE`、`num_warps` 等编译参数；
- 在线 msDebug 和异常算子 Dump 使用同一调试通道时可能冲突，不应同时开启；
- 部分优化会改变变量生命周期，即使开启变量调试信息，也不能保证每个源码变量始终可打印；
- msDebug、CANN、驱动、固件、芯片和 Triton-Ascend 需要使用相互兼容的版本。

官方环境和约束说明可参考 [msDebug 使用前准备](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0063.html?framework=mindspore)。

## 产品支持情况

官方 msDebug 当前文档列出了昇腾 A2、A3、310P 以及 950 系列等产品形态，但具体功能范围仍取决于 msDebug、驱动和 CANN 版本。

对于 Triton 场景，还必须确认当前 Triton-Ascend 后端支持目标芯片。可以通过编译日志、Kernel Dump 中的目标信息或 Triton 后端 Target 信息确认实际架构。

SIMD、SIMT、AIC、AIV、Core Dump 和局部变量打印的支持范围可能不同。本文中的功能均应以本机 msDebug 的 `help` 输出和对应版本发行说明为准。

## 命令参考

| 命令 | 缩写 | 说明 | Triton 示例 |
|---|---|---|---|
| `breakpoint set -f <file> -l <line>` | `b` | 设置源码行断点 | `b /data/demo/multibuffer.py:15` |
| `run` | `r` | 启动 Python 程序 | `r` |
| `continue` | `c` | 从断点继续运行 | `c` |
| `print <variable>` | `p` | 打印当前 PC 可用的变量 | `p x` |
| `frame variable` | `var` | 打印当前作用域局部变量 | `var` |
| `memory read` | `x` | 读取 GM、UB 等内存 | `x -m UB -f float32[] 0x600 -s 128 -c 1` |
| `thread step-over` | `n` | 单步到下一可执行源码行 | `n` |
| `thread step-in` | `s` | 尝试进入函数 | `s` |
| `thread step-out` | `finish` | 执行到当前函数返回 | `finish` |
| `register read -a` | `re r -a` | 读取当前核全部可用寄存器 | `register read -a` |
| `register read` | `re r` | 读取指定寄存器 | 见下方示例 |
| `ascend info devices` | - | 查看 Device 信息 | `ascend info devices` |
| `ascend info cores` | - | 查看核、PC、Block 和停止原因 | `ascend info cores` |
| `ascend info tasks` | - | 查看 Task 信息 | `ascend info tasks` |
| `ascend info stream` | - | 查看 Stream 信息 | `ascend info stream` |
| `ascend info blocks` | - | 查看 Block 信息 | `ascend info blocks -d` |
| `ascend aiv <id>` | - | 切换当前焦点 AIV | `ascend aiv 3` |
| `ascend aic <id>` | - | 切换当前焦点 AIC | `ascend aic 17` |
| `ascend info threads` | - | 查看 SIMT 线程 | `ascend info threads` |
| `ascend thread <id>` | - | 切换 SIMT 线程 | `ascend thread (0,8,0)` |
| `image add <kernel.o>` | - | 导入 Kernel 调试信息 | `image add /data/dump/hash/kernel.o` |
| `image load -f <kernel.o> -s <slide>` | - | 按运行时偏移加载调试信息 | `image load -f /data/dump/hash/kernel.o -s 0` |
| `thread backtrace` | `bt` | 查看调用栈，当前主要用于 Core Dump | `bt` |
| `ascend info summary` | - | 查看 Core Dump 摘要 | `ascend info summary` |
| `help <command>` | - | 查看本机命令帮助 | `help memory read` |

读取指定寄存器时，寄存器名前需添加 `$`，例如：

```text
(msdebug) register read $PC $GPR0
```

`memory read` 常用选项：

| 选项 | 含义 |
|---|---|
| `-m` | 内存类型，如 `GM`、`UB`、`L0A`、`L0B`、`L0C`、`L1` 等 |
| `-f` | 显示数据类型，如 `float16[]`、`float32[]`、`int32_t[]` |
| `-s` | 每行打印的字节数 |
| `-c` | 打印行数 |
| `-E` / `--offset` | 打印前跳过的元素数 |

不同版本的格式名称可能略有不同，使用前执行：

```text
(msdebug) help memory read
(msdebug) help x
```

## 工具使用

### 准备 Triton 调试用例

下面使用一个简单的向量加法说明。保存为 `triton_msdebug_demo.py`：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    output = x + y

    tl.store(output_ptr + offsets, output)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    n_elements = x.numel()

    output = torch.empty_like(x)

    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

if __name__ == "__main__":
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 6.0, 8.0]

    x = torch.tensor(a, device = 'npu')
    y = torch.tensor(b, device = 'npu')

    output_triton = vector_add(x, y)

    print(output_triton)

```

### 启动工具

```bash
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1
export TRITON_DISABLE_LINE_INFO=0
msdebug python triton_msdebug_demo.py
```

典型回显：

```text
=================================================================
                   >>>>>   MindStudio   <<<<<
    THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE
=================================================================

msdebug 26.0.0
(msdebug) target create "python"
Current executable set to '/usr/local/python3.11.13/bin/python' (x86_64).
(msdebug) settings set -- target.run-args  "multibuffer.py"
Cannot read termcap database;
using dumb terminal settings.
(msdebug)
```

修改了 Triton 源码、编译常量或编译器代码后，需要重新开启强制编译并重新生成二进制。

```bash
export TRITON_ALWAYS_COMPILE=1
```

### 指定 Device ID

多 Device 程序可在运行前指定调试焦点：

```text
(msdebug) ascend device 1
(msdebug) r
```

如果不指定，通常调试目标程序首次设置的 Device。调试用例应尽量固定 Device，避免目标 Kernel 被下发到另一个 Device。

### 调试退出

```text
(msdebug) q
Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] y
```

## 断点设置功能介绍

### 功能说明

msDebug 可以在 Triton Python Kernel 源码的指定行设置断点。设备 PC 与 Python 行号的对应关系来自 `kernel.o` 中的 DWARF 行表。

### 注意事项

- 建议对 `.py` 文件使用绝对路径，避免同名文件或路径改写导致错误匹配；
- 应在 `tl.load`、实际计算、`tl.store` 等会生成设备指令的语句上设置断点；
- 仅用于组织代码的 Python 语句、装饰器或被编译期消除的表达式不一定有设备地址；
- Triton JIT 模块在执行 `r` 后才装载，因此运行前断点显示 pending 通常是正常现象；
- 一条 Triton 语句可能扩展成多条地址计算、cast、memref 和 load/store 指令，断点可能解析出多个 Location；
- `debug_line_rewriter.py`、Location 归一化 Pass 和调试 NOP 会影响最终可停位置，但 msDebug 本身不会重新生成缺失的行表。

### 使用示例

#### 设置行断点

假设 `BREAK_X` 位于第 13 行：

```text
(msdebug) b triton_msdebug_demo.py:13
Breakpoint 1: no locations (pending on future shared library load).
WARNING:  Unable to resolve breakpoint to any actual locations.
(msdebug) r
```

Kernel 完成 JIT 装载后，预期出现类似信息：

```text
[Launch of Kernel add_kernel ...]
1 location added to breakpoint 1
Process ... stopped
[Switching to focus on Kernel add_kernel, CoreId ..., Type aiv]
```

pending 的含义是“当前尚未装载包含该源码行的设备模块”，不是立即失败。

#### 查看断点

```text
(msdebug) breakpoint list
```

#### 删除断点

```text
(msdebug) breakpoint delete 1
```

## 内存与变量打印功能介绍

### 功能说明

Triton 源码中的变量可能位于寄存器、UB 或 GM，也可能只是编译器 SSA 值，没有固定内存地址。msDebug 可以在断点停止后打印当前 PC 可用的变量，并读取当前焦点核的内存。

### 注意事项

- `p x` 成功需要 `kernel.o` 同时包含变量 DIE 和当前 PC 有效的 Location；
- 源码断点能命中，只能说明 `.debug_line` 有效，不能说明变量 Location 一定存在；
- UB 是核内局部存储，切换 AIV/AIC 后，相同 UB 偏移对应另一核的数据；
- GM 使用设备全局地址，读取前必须确认地址和长度合法；
- 编译器可能把 Triton 变量保存在寄存器中，变量不一定拥有 UB 地址；
- 寄存器和 UB 会在变量最后一次使用后复用；
- `tl.load` 的 mask 为 false 时，部分 lane 的数据可能无效；
- `-f`内存显示格式必须与真实元素类型一致,不指定默认十六进制表示；
- 不使用`-c`指定打印行数默认为打印`4`行，不使用`-s`指定每行打印字节数默认为`128`字节。

### 使用示例

#### 打印指定变量

```text
(msdebug) p x
(msdebug) p y
(msdebug) p z
```

若变量信息可用，msDebug 会显示变量值、地址或其所在位置；如果显示 `<unavailable>`，应判断当前 PC 是否处于变量有效范围。

#### 打印所有局部变量

```text
(msdebug) var
```

Triton 编译后的变量名可能与 Python 源码名不同，也可能只保留部分变量。

#### 按 Triton 变量读取 UB

在支持变量名解析的版本中，可以使用：

```text
(msdebug) x -m UB -f float32 x
(msdebug) x -m UB -f float32 z -s 32
```

如果 `x` 对应一组 128 个 `float32` 元素，`-s 32` 表示每行显示 32 字节，即每行 8 个 `float32`；因此可能看到 4 行或更多行，具体总长度由变量调试信息和命令参数共同决定。

#### 按已知 UB 地址读取

```text
(msdebug) x -m UB -f float32[] 0x600 -s 128 -c 1
```

该命令从当前焦点核的 UB 偏移 `0x600` 开始，按 `float32` 解释数据，每行打印 128 字节，共打印 1 行，即最多显示 32 个 `float32` 元素。

#### 按已知 GM 地址读取

```text
(msdebug) x -m GM -f float32[] 0x00001240c0037000 -s 128 -c 2
```

该命令从指定 GM 地址开始，按 `float32` 格式打印两行，每行 128 字节。

若需要从起始地址跳过若干元素：

```text
(msdebug) x -m GM -f float32[] 0x00001240c0037000 -s 128 -c 1 -E 16
```

#### 处理默认读取上限

当读取超过默认上限时，msDebug 可能提示：

```text
Normally, 'memory read' will not read over 1024 bytes of data.
```

仅本次强制读取时，按本机 `help memory read` 中的 `--force` 选项执行；需要多次读取时，可调整：

```text
(msdebug) settings set target.max-memory-read-size 4096
```

修改上限前应先验证地址、元素类型和读取范围。

#### `x`、`y`、`z` 地址与 PC 生命周期

以下是一个典型而非绝对的生命周期：

| 当前停点 | `x` | `y` | `z` |
|---|---|---|---|
| `tl.load(x)` 执行前 | 尚未产生 | 尚未产生 | 尚未产生 |
| `tl.load(x)` 执行后 | 可能位于寄存器或 UB | 尚未产生 | 尚未产生 |
| `tl.load(y)` 执行后 | 可能可用 | 可能可用 | 尚未产生 |
| `z = x + y` 执行后 | 可能仍可用，也可能开始复用 | 可能仍可用，也可能开始复用 | 可能位于寄存器或 UB |
| `tl.store(z)` 执行后 | 可能失效 | 可能失效 | 可能失效或只剩 GM 输出 |

建议在多个连续停点分别执行：

```text
(msdebug) n
(msdebug) p x
(msdebug) p y
(msdebug) p z
(msdebug) register read -a
```

变量无法打印时，离线检查：

```bash
llvm-dwarfdump --debug-info "$KERNEL_O" | less
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
llvm-dwarfdump --debug-loclists "$KERNEL_O" | less
```

重点查看：

- 是否有对应 `DW_TAG_variable`；
- 是否有 `DW_AT_location`；
- Location List 的 PC 范围是否覆盖当前断点 PC；
- Location 是寄存器、UB 地址还是其他表达式；
- 同名变量是否属于另一个编译单元或词法作用域。

如果变量 DIE 存在但没有 `DW_AT_location`，msDebug 知道变量名称，却无法恢复它当前存放的位置，此时按变量名打印失败是预期结果。

官方内存命令和各选项说明可参考 [内存与变量打印](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0067.html?framework=mindspore)。

## 单步调试功能介绍

### 功能说明

命中源码断点后，可以使用 `n`、`s` 和 `finish` 控制当前焦点核继续执行。

### 注意事项

- Triton Python 函数经过内联和多级 Lowering，`s` 不一定能进入 Python 层函数；
- 一条 Python 语句可能对应多条设备指令，同一源码行可能停止多次；
- 多条内部 IR 也可能统一映射到同一源码行；
- 被优化掉或没有独立设备指令的行会被跳过；
- load/store 扩展出的 cast、memref、地址计算和调试 NOP 会影响单步停点；
- 单步针对当前带 `*` 的焦点核，其他核可能仍停在原断点；
- 如果 Kernel 使用同步指令，单独推进一个核可能表现为等待，应结合所有核的 PC 判断。

### 使用示例

#### Step Over

```text
(msdebug) b triton_msdebug_demo.py:<BREAK_X行号>
(msdebug) r
(msdebug) n
```

执行后查看所有核状态：

```text
(msdebug) ascend info cores
```

当前核的 `stop reason` 应显示 `step over` 或仍显示同时命中的 breakpoint。

#### Step In

```text
(msdebug) s
```

如果后端已将函数内联，`s` 可能表现为移动到同一函数中的下一个有效 Location，而不是进入新的 Python 栈帧。

#### Step Out

```text
(msdebug) finish
```

对于完全内联的 Triton Kernel，可能不存在可见的独立调用帧，此命令的效果取决于最终 DWARF 调用信息。

#### 判断是重复执行还是行号重复

如果 `n` 多次停在同一源码行，结合当前 PC 和反汇编判断：

```text
(msdebug) re r $PC
(msdebug) disassemble
```

同时离线查看：

```bash
llvm-dwarfdump --debug-line "$KERNEL_O"
```

只要 PC 在变化，通常说明调试器正在执行映射到同一源码行的不同设备指令，并不代表 Python 语句被重复运行。

## 中断运行功能介绍

### 功能说明

当 Triton Kernel 长时间不返回时，可通过 `Ctrl+C` 中断在 msDebug 内启动的程序，查看当前停止位置。

### 注意事项

- 只能中断由当前 msDebug 会话启动的程序；
- 卡住可能来自 Kernel 死循环、核间同步、等待事件或错误的控制流；
- 根据官方当前约束，`Ctrl+C` 中断后主要支持调试信息展示和核切换；单步、寄存器读取、变量/内存打印和 `continue` 可能不可用；
- 因此，`Ctrl+C` 主要用于确认各核卡在什么位置，而不是替代普通断点调试。

### 使用示例

```text
(msdebug) r
<按 Ctrl+C>
(msdebug) ascend info cores
(msdebug) ascend info blocks -d
```

观察是否出现：

- 某个核 PC 长时间不变；
- 多个核停在同步或等待位置；
- 某个 Block 未到达与其他 Block 相同的阶段；
- 当前源码位置落在 Triton 循环或 load/store 等待附近。

完成信息收集后退出：

```text
(msdebug) q
```

## 核切换功能介绍

### 功能说明

将调试焦点切换到指定 AIV 或 AIC。切换后，源码位置、寄存器以及 UB 等局部内存操作都针对新焦点核。

### 使用示例

查看当前核：

```text
(msdebug) ascend info cores
```

切换到 AIV 3：

```text
(msdebug) ascend aiv 3
```

切换到 AIC 17：

```text
(msdebug) ascend aic 17
```

再次确认焦点：

```text
(msdebug) ascend info cores
```

输出中带 `*` 的行是当前焦点核。

### Triton 多 Block 场景说明

Triton 的 `program_id`/Block 会被调度到设备核执行。查询映射：

```text
(msdebug) ascend info blocks
(msdebug) ascend info blocks -d
```

读取分核数据前，应先执行：

```text
(msdebug) ascend aiv <core-id>
(msdebug) p x
(msdebug) x -m UB -f float32 x
```

切换到另一个 AIV 后再次读取，即可比较不同 Block/核的数据。两个核即使都显示 `UB 0x600`，也代表各自核的局部 UB 数据，不是同一物理上下文。

官方核切换示例可参考 [Vector 算子调试示例](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0074.html?framework=mindspore)。

## 检查程序状态功能介绍

### 功能说明

在普通断点或单步停点，可以读取当前焦点核的寄存器，确认 PC、通用寄存器及状态寄存器。

### 使用示例

读取全部可用寄存器：

```text
(msdebug) register read -a
```

读取指定寄存器：

```text
(msdebug) register read $PC $GPR0 $GPR30
```

也可使用缩写：

```text
(msdebug) re r $PC
```

对 Triton 问题，优先关注：

- `PC`：与 `.debug_line`、反汇编及变量 Location List 对照；
- 通用寄存器：分析地址计算、循环索引或指针；
- 状态和事件寄存器：分析 Kernel 等待或同步问题；
- 切核后的寄存器差异：判断是否只有单个 Block/核异常。

寄存器名无效时会返回相应错误，应使用 `register read -a` 确认当前芯片和核类型实际支持的名称。

## 调试信息展示功能介绍

### 功能说明

查询当前 Kernel 所在的 Device、Stream、Task、Block、核和停止原因。

### 使用示例

#### Device 信息

```text
(msdebug) ascend info devices
```

`*` 表示当前焦点 Device。常见字段包括 Device ID、AIC/AIV 数量和核 Mask。

#### 核信息

```text
(msdebug) ascend info cores
```

重点字段：

| 字段 | Triton 调试含义 |
|---|---|
| `CoreId` | 当前 AIC/AIV 的核 ID |
| `Type` | 核类型，通常为 `aic` 或 `aiv` |
| `Device` | 逻辑 Device ID |
| `Stream` | Kernel 下发所在 Stream |
| `Task` | 当前 Stream 中的 Task |
| `Block` | Triton program/Block 对应的逻辑 Block ID |
| `PC` | 当前核设备 PC |
| `stop reason` | breakpoint、step over、step in、Ctrl+C 或异常类型等 |
| `Filename/Line` | PC 对应的 Triton 源文件和行号，无法映射时可能为 `NA` |

#### Task 信息

```text
(msdebug) ascend info tasks
```

用于确认当前 Task 对应的 Kernel Invocation，特别适合一个 Python 脚本调用多个 Kernel 时筛选目标。

#### Stream 信息

```text
(msdebug) ascend info stream
```

用于确认目标 Kernel 所在的 Stream 和核类型。

#### Block 信息

```text
(msdebug) ascend info blocks
(msdebug) ascend info blocks -d
```

`-d` 会显示所有 Block 当前停止位置，适合排查多核不同步、某个 Block 路径异常以及分核数据差异。

## SIMT 线程切换功能介绍

### 功能说明

如果目标 Triton Kernel 在当前芯片和后端上以受 msDebug 支持的 SIMT 方式执行，可以查询并切换 SIMT 线程。SIMD Kernel 或不支持 SIMT 调试的版本不适用本节。

### 使用示例

查询线程：

```text
(msdebug) ascend info threads
```

查询或切换线程：

```text
(msdebug) ascend thread 100
(msdebug) ascend thread (0,8,0)
```

切换线程后，源码位置和变量上下文会变化。应重新确认：

```text
(msdebug) ascend info threads
(msdebug) var
```

若命令不受支持或没有线程信息，应先确认 Kernel 编程模式、芯片和 msDebug 版本，而不是把它判断为 Triton 行号问题。

## 解析异常算子 Dump 文件功能介绍

### 功能说明

NPU 发生硬件异常时，可以生成异常算子 Core Dump，离线恢复异常核、PC、寄存器和部分内存现场。此功能适用于在线断点无法稳定复现或程序已经异常退出的场景。

### 注意事项

- 开启异常算子 Dump 后，不应同时使用在线 msDebug；
- 必须尽量使用异常现场完全匹配的 `kernel.o`；
- Python 源码相同不代表二进制相同，Block Size、Autotune、编译器版本和后端选项都会改变二进制；
- 硬件可能在产生异常后继续执行若干指令再上报，因此 Core 中部分内存和寄存器可能不是最初故障瞬间的值；
- 异常 PC 通常比普通数据更可靠，应先从 PC、停止原因和反汇编开始；
- 官方当前说明中，`bt` 主要用于 Core Dump，且只在部分硬件异常停止原因下保证准确性。

### 使用示例

#### 生成异常算子 Core 文件

在线 msDebug 退出后，在普通终端设置：

```bash
export ASCEND_DUMP_SCENE=aic_err_detail_dump
export ASCEND_DUMP_PATH=/data/ascend_core_dump
mkdir -p "$ASCEND_DUMP_PATH"
python triton_msdebug_demo.py
```

只有触发相应设备异常时才会生成 `.core` 文件。

#### 加载 Core 文件和 `kernel.o`

```bash
msdebug --core /data/ascend_core_dump/<error>.core \
  /data/triton_msdebug_dump/<hash>/kernel.o
```

如果暂时没有二进制，也可以只加载 Core：

```bash
msdebug --core /data/ascend_core_dump/<error>.core
```

但缺少匹配 `kernel.o` 时，源码行、符号和调用栈解析会受到限制。

#### 查看异常摘要

```text
(msdebug) ascend info summary
(msdebug) ascend info devices
(msdebug) ascend info cores
(msdebug) ascend info blocks -d
```

#### 切换异常核

```text
(msdebug) ascend aiv <id>
(msdebug) ascend aic <id>
```

选择 `stop reason` 为 `MTE_ERROR`、`VEC_ERROR`、`CUBE_ERROR` 等异常的核，然后查看：

```text
(msdebug) bt
(msdebug) frame select 0
(msdebug) register read -a
```

#### 根据 PC 定位指令

```text
(msdebug) re r $PC
(msdebug) image list -o -f
(msdebug) image lookup -a <pc-address>
(msdebug) disassemble -s <start-address> -e <end-address>
```

#### 读取 Core 中内存

根据 `ascend info summary` 输出的内存段和当前核读取：

```text
(msdebug) x -m GM -f uint8_t[] <address> -s 128 -c 1
(msdebug) x -m STACK -f uint8_t[] <address> -s 128 -c 1
(msdebug) x -m DCACHE -f uint8_t[] <address> -s 128 -c 1
```

具体支持的 Core Dump 内存类型应以 `help memory read` 为准。

## 常见问题

### msDebug 初始化失败并提示 `0x20102`

典型信息：

```text
msdebug failed to initialize, please run "echo 1 > /proc/debug_switch"
```

检查：

```bash
echo 1 > /proc/debug_switch
cat /proc/debug_switch
ls -l /dev/drv_debug
```

容器中无法修改时应回到宿主机处理。

### 断点命中，但 `var` 没有变量

断点使用 `.debug_line`，变量使用 `.debug_info` 和 `.debug_loc*`。按以下顺序检查：

```bash
echo "$LLVM_EXTRACT_DI_LOCAL_VARIABLES"
echo "$TRITON_DISABLE_LINE_INFO"
llvm-dwarfdump --verify "$KERNEL_O"
llvm-dwarfdump --debug-info "$KERNEL_O" | less
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
```

如果变量存在但没有 Location，或当前 PC 不在 Location List 范围内，msDebug 无法打印该变量。

### 可以用 msDebug 给 `NormalizeDebugLineLocations.cpp` 下断点吗

不可以。它是主机侧 MLIR Pass，应使用编译日志、Pass 前后 IR、GDB/LLDB 调试 `triton-opt`。msDebug 调试的是已经装载到 NPU 的 `kernel.o`。

### 读出的 UB 数据不符合预期

依次检查：

1. `ascend info cores` 中的焦点核是否正确；
2. 是否在 `tl.load` 完成后、内存复用前停止；
3. 数据类型和 `-f` 是否一致；
4. `-s` 是否按字节理解；
5. mask 是否使部分 lane 无效；
6. Block Size 和实际向量长度是否一致；
7. 是否正在调试另一份 Autotune Kernel；
8. 地址是 UB 偏移还是 GM 绝对地址。

### `LD_PRELOAD detected` 是否需要处理

该警告单独出现时通常不是失败原因。如果 msDebug 后续可以初始化、命中断点并读取设备信息，可以先保留当前环境。只有发生库冲突、符号加载失败或进程初始化失败时，再核对 `LD_PRELOAD` 指向的库。

## 附录：`kernel.o` 调试信息检查

### 检查 Section

```bash
readelf -S "$KERNEL_O" | grep -E 'debug|symtab|strtab'
```

### 校验 DWARF

```bash
llvm-dwarfdump --verify "$KERNEL_O"
```

### 查看行表

```bash
llvm-dwarfdump --debug-line "$KERNEL_O" | less
```

### 查看编译单元和变量

```bash
llvm-dwarfdump --debug-info "$KERNEL_O" | less
```

### 查看变量位置

```bash
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
llvm-dwarfdump --debug-loclists "$KERNEL_O" | less
```

### 保存分析结果

```bash
llvm-dwarfdump --verify "$KERNEL_O" > dwarf-verify.txt 2>&1
llvm-dwarfdump --debug-info "$KERNEL_O" > dwarf-info.txt
llvm-dwarfdump --debug-line "$KERNEL_O" > dwarf-line.txt
llvm-dwarfdump --debug-loc "$KERNEL_O" > dwarf-locations.txt
```

提交 Triton-Ascend 调试问题时，建议同时提供最小 Python 用例、版本信息、环境变量、完全匹配的 `kernel.o`、上述 DWARF 输出、msDebug 会话日志、目标 Device/核/Block 和实际停止 PC。

## 参考资料

- [MindStudio Debugger 用户指南](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)
- [MindStudio Debugger 工具概述](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0062.html?framework=mindspore)
- [Cuda-gdb官方文档](https://docs.nvidia.com/cuda/cuda-gdb/index.html#single-stepping)
- [DWARF location list 说明](https://dwarfstd.org/doc/040408.1.html)
- [LLVM 调试变量说明](https://llvm.org/docs/SourceLevelDebugging.html)
