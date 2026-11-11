# Triton msDebug User Guide

## Introduction

msDebug debugs device programs running on Ascend NPUs. It supports source breakpoints, stepping, variable and memory inspection, register reads, core switching, execution-state queries, and analysis of core dumps from failed operators.

Use msDebug to investigate:

- Incorrect Triton kernel results.
- Unexpected data during `tl.load`, computation, or `tl.store`.
- Incorrect results on specific blocks or cores in a multi-block or multi-core workload.
- A kernel that hangs, when you need to locate where execution stopped.
- NPU hardware exceptions such as MTE, VEC, or CUBE errors.
- Exception state saved in a `.core` file.
- Problems with source breakpoints, line mappings, or variable debug information.

msDebug does not debug Triton compiler passes themselves. If compilation fails before `kernel.o` is generated, use IR dumps, compiler logs, and GDB/LLDB to debug `triton-opt` or the compiler process.

## Preparation

### Environment Setup

- Install msDebug by following the [msDebug Installation Guide](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/install_guide/msdebug_install_guide.md).
- Enable msDebug by installing the NPU driver and firmware using either method below. Use a CANN version later than 8.1.RC1 and a driver version later than 25.0.RC1. Method 1 is recommended.

**Method 1**: Install the driver with `--full`, then run `echo 1 > /proc/debug_switch` as root to enable the debug channel.

```bash
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full
```

**Method 2**: Install the driver with `--debug`. See [Installing the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0107.html) in the CANN Software Installation Guide for details.

```bash
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --debug
```

### Generating Debug Information for a Triton Kernel

To enable msDebug support in Triton-Ascend, set the following environment variables:

```bash
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1
export TRITON_DISABLE_LINE_INFO=0
```

| Environment Variable | Purpose |
|---|---|
| `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` | Enables local-variable debugging in the Ascend backend, including the associated location, NOP, and line-table processing |
| `TRITON_DISABLE_LINE_INFO=0` | Enables source line information; set it explicitly because the backend may disable it by default |

The current implementation passes `--enable-debug-info=true` and `--enable-debug-variables=true` to the backend compiler.

### Constraints

- The debug channel requires elevated privileges. Avoid leaving it enabled in production.
- Run only one msDebug session per device and pause other operator tasks.
- Prefer a Python program that invokes only the target Triton kernel.
- For initial troubleshooting, disable autotuning and use fixed compilation parameters such as `BLOCK_SIZE` and `num_warps`.
- Online msDebug sessions and exception dumps may conflict when using the same debug channel. Do not enable both at once.
- Optimizations can change variable lifetimes. Enabling variable debug information does not guarantee that every source variable is available at every stop.
- Use compatible versions of msDebug, CANN, the driver, firmware, hardware, and Triton-Ascend.

For an official introduction, see the [msDebug Quick Start](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msdebug_quick_start.md).

## Supported Products

The official msDebug documentation lists Ascend A2, A3, 310P, and 950 series products. The available features depend on the msDebug, driver, and CANN versions.

For Triton workloads, also confirm that the Triton-Ascend backend supports the target chip. Check the architecture in compiler logs, kernel dump metadata, or the Triton backend target information.

Support may differ for SIMD, SIMT, AIC, AIV, core dumps, and local-variable inspection. Refer to the installed msDebug version's `help` output and release notes for the features described in this guide.

## Command Reference

| Command | Alias | Description | Triton Example |
|---|---|---|---|
| `breakpoint set -f <file> -l <line>` | `b` | Set a source-line breakpoint | `b /data/demo/multibuffer.py:15` |
| `run` | `r` | Start the Python program | `r` |
| `continue` | `c` | Resume execution from a breakpoint | `c` |
| `print <variable>` | `p` | Print a variable available at the current PC | `p x` |
| `frame variable` | `var` | Print local variables in the current scope | `var` |
| `memory read` | `x` | Read memory such as GM or UB | `x -m UB -f float32[] 0x600 -s 128 -c 1` |
| `thread step-over` | `n` | Step to the next executable source line | `n` |
| `thread step-in` | `s` | Attempt to step into a function | `s` |
| `thread step-out` | `finish` | Run until the current function returns | `finish` |
| `register read -a` | `re r -a` | Read all available registers on the current core | `register read -a` |
| `register read` | `re r` | Read specific registers | See the example below |
| `ascend info devices` | - | Show device information | `ascend info devices` |
| `ascend info cores` | - | Show cores, PCs, blocks, and stop reasons | `ascend info cores` |
| `ascend info tasks` | - | Show task information | `ascend info tasks` |
| `ascend info stream` | - | Show stream information | `ascend info stream` |
| `ascend info blocks` | - | Show block information | `ascend info blocks -d` |
| `ascend aiv <id>` | - | Switch focus to an AIV core | `ascend aiv 3` |
| `ascend aic <id>` | - | Switch focus to an AIC core | `ascend aic 17` |
| `ascend info threads` | - | Show SIMT threads | `ascend info threads` |
| `ascend thread <id>` | - | Switch SIMT threads | `ascend thread (0,8,0)` |
| `image add <kernel.o>` | - | Import kernel debug information | `image add /data/dump/hash/kernel.o` |
| `image load -f <kernel.o> -s <slide>` | - | Load debug information with a runtime offset | `image load -f /data/dump/hash/kernel.o -s 0` |
| `thread backtrace` | `bt` | Show the call stack, currently mainly for core dumps | `bt` |
| `ascend info summary` | - | Show a core dump summary | `ascend info summary` |
| `help <command>` | - | Show help for the installed version | `help memory read` |

Prefix register names with `$` when reading specific registers:

```text
(msdebug) register read $PC $GPR0
```

Common `memory read` options:

| Option | Meaning |
|---|---|
| `-m` | Memory type, such as `GM`, `UB`, `L0A`, `L0B`, `L0C`, or `L1` |
| `-f` | Display data type, such as `float16[]`, `float32[]`, or `int32_t[]` |
| `-s` | Bytes printed per row |
| `-c` | Number of rows to print |
| `-E` / `--offset` | Number of elements to skip before printing |

Format names may differ between versions. Check the installed version before use:

```text
(msdebug) help memory read
(msdebug) help x
```

## Using the Tool

### Preparing a Triton Debugging Example

The following vector addition illustrates the workflow. Save it as `triton_msdebug_demo.py`:

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

### Starting the Tool

```bash
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1
export TRITON_DISABLE_LINE_INFO=0
msdebug python triton_msdebug_demo.py
```

Typical output:

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

After changing Triton source code, compile-time constants, or compiler code, force recompilation to regenerate the binary:

```bash
export TRITON_ALWAYS_COMPILE=1
```

### Selecting a Device ID

For a program using multiple devices, select the debugging focus before starting execution:

```text
(msdebug) ascend device 1
(msdebug) r
```

If no device is specified, debugging normally targets the first device selected by the program. Keep the device fixed in the example so that the target kernel is not dispatched to a different device.

### Exiting the Debugger

```text
(msdebug) q
Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] y
```

## Setting Breakpoints

### Overview

msDebug can set breakpoints on specific lines of Triton Python kernel source code. The mapping between device PCs and Python line numbers comes from the DWARF line table in `kernel.o`.

### Notes

- Use absolute paths for `.py` files to avoid mismatches caused by duplicate filenames or path rewriting.
- Set breakpoints on statements that generate device instructions, such as `tl.load`, computations, and `tl.store`.
- Python statements used only to organize code, decorators, and expressions eliminated at compile time may have no device address.
- Triton JIT modules are loaded after `r` starts execution, so pending breakpoints before execution are usually normal.
- A single Triton statement can expand into multiple address calculations, casts, memref operations, and load/store instructions. A breakpoint may resolve to multiple locations.
- `debug_line_rewriter.py`, location normalization passes, and debug NOPs affect where execution can stop. msDebug does not regenerate missing line tables.

### Examples

#### Setting a Line Breakpoint

Assume the line marked `BREAK_X` is line 13:

```text
(msdebug) b triton_msdebug_demo.py:13
Breakpoint 1: no locations (pending on future shared library load).
WARNING:  Unable to resolve breakpoint to any actual locations.
(msdebug) r
```

After the JIT kernel is loaded, expect output similar to:

```text
[Launch of Kernel add_kernel ...]
1 location added to breakpoint 1
Process ... stopped
[Switching to focus on Kernel add_kernel, CoreId ..., Type aiv]
```

A pending breakpoint means that the device module containing the source line has not yet been loaded; it does not indicate an immediate failure.

#### Listing Breakpoints

```text
(msdebug) breakpoint list
```

#### Deleting a Breakpoint

```text
(msdebug) breakpoint delete 1
```

## Inspecting Memory and Variables

### Overview

A Triton source variable may reside in registers, UB, or GM, or exist only as a compiler SSA value with no fixed memory address. At a breakpoint, msDebug can print variables available at the current PC and read memory for the core in focus.

### Notes

- For `p x` to succeed, `kernel.o` must contain both the variable DIE and a location valid at the current PC.
- Hitting a source breakpoint confirms that `.debug_line` works, but does not guarantee that variable locations exist.
- UB is local to a core. After switching AIV/AIC cores, the same UB offset refers to another core's data.
- GM uses device-global addresses. Verify the address and length before reading.
- The compiler may keep Triton variables in registers, so a variable may have no UB address.
- Registers and UB can be reused after a variable's last use.
- Some lanes may contain invalid data when the `tl.load` mask is false.
- The `-f` format must match the actual element type. The default display is hexadecimal.
- Without `-c`, the default is `4` rows. Without `-s`, the default is `128` bytes per row.

### Examples

#### Printing Specific Variables

```text
(msdebug) p x
(msdebug) p y
(msdebug) p z
```

When variable information is available, msDebug displays the value, address, or storage location. If it shows `<unavailable>`, check whether the current PC is within the variable's valid range.

#### Printing All Local Variables

```text
(msdebug) var
```

Compiled Triton variable names may differ from Python source names, and some variables may not be retained.

#### Reading UB Using a Triton Variable

On versions that support variable-name resolution, use:

```text
(msdebug) x -m UB -f float32 x
(msdebug) x -m UB -f float32 z -s 32
```

If `x` represents 128 `float32` elements, `-s 32` prints 32 bytes per row, or eight `float32` values. You may see four or more rows; the total amount depends on the variable debug information and command arguments.

#### Reading a Known UB Address

```text
(msdebug) x -m UB -f float32[] 0x600 -s 128 -c 1
```

This reads from UB offset `0x600` on the core in focus, interpreting the data as `float32`. It prints one row of 128 bytes, or at most 32 `float32` elements.

#### Reading a Known GM Address

```text
(msdebug) x -m GM -f float32[] 0x00001240c0037000 -s 128 -c 2
```

This prints two rows of 128 bytes from the specified GM address in `float32` format.

To skip elements from the starting address, use:

```text
(msdebug) x -m GM -f float32[] 0x00001240c0037000 -s 128 -c 1 -E 16
```

#### Handling the Default Read Limit

When a read exceeds the default limit, msDebug may report:

```text
Normally, 'memory read' will not read over 1024 bytes of data.
```

For a single forced read, use `--force` as described by the installed version's `help memory read`. For repeated reads, adjust the limit:

```text
(msdebug) settings set target.max-memory-read-size 4096
```

Verify the address, element type, and read range before increasing the limit.

#### Addresses and PC Lifetimes of `x`, `y`, and `z`

The following is a typical, but not guaranteed, lifetime:

| Stop Position | `x` | `y` | `z` |
|---|---|---|---|
| Before `tl.load(x)` | Not yet produced | Not yet produced | Not yet produced |
| After `tl.load(x)` | May be in registers or UB | Not yet produced | Not yet produced |
| After `tl.load(y)` | May be available | May be available | Not yet produced |
| After `z = x + y` | May still be available, or storage may be reused | May still be available, or storage may be reused | May be in registers or UB |
| After `tl.store(z)` | May be unavailable | May be unavailable | May be unavailable, or only the GM output may remain |

At successive stops, try:

```text
(msdebug) n
(msdebug) p x
(msdebug) p y
(msdebug) p z
(msdebug) register read -a
```

If variables cannot be printed, inspect the debug information offline:

```bash
llvm-dwarfdump --debug-info "$KERNEL_O" | less
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
llvm-dwarfdump --debug-loclists "$KERNEL_O" | less
```

Check:

- Whether the corresponding `DW_TAG_variable` exists.
- Whether it has a `DW_AT_location`.
- Whether the location list's PC range includes the current breakpoint PC.
- Whether the location describes a register, UB address, or another expression.
- Whether a variable with the same name belongs to another compilation unit or lexical scope.

If a variable DIE exists but has no `DW_AT_location`, msDebug knows the name but cannot recover its current storage location. Printing that variable by name is expected to fail.

For official memory commands and options, see [Memory and Variable Inspection](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msdebug_user_guide.md#%E5%86%85%E5%AD%98%E4%B8%8E%E5%8F%98%E9%87%8F%E6%89%93%E5%8D%B0%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D).

## Single-Stepping

### Overview

After hitting a source breakpoint, use `n`, `s`, and `finish` to control execution on the core in focus.

### Notes

- Triton Python functions undergo inlining and multiple lowering stages. `s` may not enter a Python-level function.
- One Python statement may map to several device instructions, so execution may stop on the same source line more than once.
- Multiple internal IR operations may also map to the same source line.
- Lines that are optimized away or have no separate device instructions are skipped.
- Casts, memref operations, address calculations, and debug NOPs introduced for loads and stores affect stepping locations.
- Stepping applies to the core marked with `*`; other cores may remain at the original breakpoint.
- If the kernel uses synchronization instructions, advancing only one core can cause it to wait. Inspect the PCs of all cores.

### Examples

#### Step Over

```text
(msdebug) b triton_msdebug_demo.py:<BREAK_X-line-number>
(msdebug) r
(msdebug) n
```

Inspect all cores after stepping:

```text
(msdebug) ascend info cores
```

The current core's `stop reason` should show `step over`, or a breakpoint hit at the same location.

#### Step In

```text
(msdebug) s
```

If the backend has inlined the function, `s` may advance to the next valid location in the same function instead of entering a new Python stack frame.

#### Step Out

```text
(msdebug) finish
```

A fully inlined Triton kernel may have no separate visible call frame. The result depends on the final DWARF call information.

#### Distinguishing Repeated Execution from Repeated Line Numbers

If `n` repeatedly stops on the same source line, inspect the PC and disassembly:

```text
(msdebug) re r $PC
(msdebug) disassemble
```

Also inspect the line table offline:

```bash
llvm-dwarfdump --debug-line "$KERNEL_O"
```

A changing PC usually means that the debugger is executing different device instructions mapped to the same source line, rather than executing the Python statement repeatedly.

## Interrupting Execution

### Overview

If a Triton kernel does not return for a long time, press `Ctrl+C` to interrupt the program launched in msDebug and inspect where it stopped.

### Notes

- You can only interrupt a program launched by the current msDebug session.
- A hang may be caused by an infinite loop, inter-core synchronization, an event wait, or incorrect control flow.
- Under the documented constraints, after `Ctrl+C`, support is mainly limited to displaying debug information and switching cores. Stepping, register reads, variable/memory inspection, and `continue` may be unavailable.
- Use `Ctrl+C` primarily to locate where cores are stuck, rather than as a substitute for normal breakpoint debugging.

### Example

```text
(msdebug) r
<press Ctrl+C>
(msdebug) ascend info cores
(msdebug) ascend info blocks -d
```

Look for:

- A core whose PC remains unchanged.
- Multiple cores stopped at synchronization or wait instructions.
- A block that has not reached the same stage as other blocks.
- A source position near a Triton loop or a load/store wait.

Exit after collecting the information:

```text
(msdebug) q
```

## Switching Cores

### Overview

Switch the debugging focus to a specific AIV or AIC core. Source positions, registers, and local-memory operations such as UB reads then refer to the newly selected core.

### Examples

List the current cores:

```text
(msdebug) ascend info cores
```

Switch to AIV 3:

```text
(msdebug) ascend aiv 3
```

Switch to AIC 17:

```text
(msdebug) ascend aic 17
```

Confirm the focus:

```text
(msdebug) ascend info cores
```

The row marked with `*` identifies the core in focus.

### Triton Multi-Block Workloads

Triton `program_id` values/blocks are scheduled onto device cores. Query the mapping with:

```text
(msdebug) ascend info blocks
(msdebug) ascend info blocks -d
```

Before reading per-core data, run:

```text
(msdebug) ascend aiv <core-id>
(msdebug) p x
(msdebug) x -m UB -f float32 x
```

Switch to another AIV and read again to compare data across blocks or cores. Even if both cores display `UB 0x600`, each address refers to that core's local UB data, not a shared physical context.

For official guidance, see [Core Switching](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msdebug_user_guide.md#%E6%A0%B8%E5%88%87%E6%8D%A2%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D).

## Inspecting Program State

### Overview

At a normal breakpoint or a stepping stop, read the registers of the core in focus to inspect the PC, general-purpose registers, and status registers.

### Examples

Read all available registers:

```text
(msdebug) register read -a
```

Read specific registers:

```text
(msdebug) register read $PC $GPR0 $GPR30
```

Or use the alias:

```text
(msdebug) re r $PC
```

For Triton issues, focus on:

- `PC`: Compare it with `.debug_line`, disassembly, and variable location lists.
- General-purpose registers: Inspect address calculations, loop indices, and pointers.
- Status and event registers: Investigate kernel waits and synchronization.
- Register differences after switching cores: Determine whether only one block or core is affected.

An invalid register name produces an error. Use `register read -a` to check which names the current chip and core type support.

## Displaying Debug Information

### Overview

Query the current kernel's device, stream, task, block, core, and stop reason.

### Examples

#### Device Information

```text
(msdebug) ascend info devices
```

`*` identifies the device in focus. Common fields include the device ID, AIC/AIV counts, and core mask.

#### Core Information

```text
(msdebug) ascend info cores
```

Key fields:

| Field | Meaning for Triton Debugging |
|---|---|
| `CoreId` | ID of the current AIC/AIV core |
| `Type` | Core type, usually `aic` or `aiv` |
| `Device` | Logical device ID |
| `Stream` | Stream on which the kernel was dispatched |
| `Task` | Task in the current stream |
| `Block` | Logical block ID corresponding to the Triton program/block |
| `PC` | Current device PC for the core |
| `stop reason` | Breakpoint, step over, step in, Ctrl+C, exception type, or another reason |
| `Filename/Line` | Triton source file and line mapped to the PC; may be `NA` if no mapping is available |

#### Task Information

```text
(msdebug) ascend info tasks
```

Use this to identify the kernel invocation associated with the current task, especially when a Python script calls multiple kernels.

#### Stream Information

```text
(msdebug) ascend info stream
```

Use this to identify the target kernel's stream and core type.

#### Block Information

```text
(msdebug) ascend info blocks
(msdebug) ascend info blocks -d
```

`-d` shows the current stop position of every block. It helps investigate cores that are out of sync, an unexpected execution path in one block, and differences in per-core data.

## Switching SIMT Threads

### Overview

If the target Triton kernel executes in a SIMT mode supported by msDebug on the current chip and backend, you can query and switch SIMT threads. This section does not apply to SIMD kernels or versions without SIMT debugging support.

### Examples

List threads:

```text
(msdebug) ascend info threads
```

Query or switch threads:

```text
(msdebug) ascend thread 100
(msdebug) ascend thread (0,8,0)
```

Switching threads changes the source position and variable context. Check them again:

```text
(msdebug) ascend info threads
(msdebug) var
```

If the command is unsupported or no thread information is available, first check the kernel programming model, chip, and msDebug version before treating it as a Triton line-mapping problem.

## Analyzing Exception Core Dumps

### Overview

An NPU hardware exception can generate an operator core dump for offline inspection of the affected core, PC, registers, and part of memory. This is useful when online breakpoints cannot reliably reproduce the issue or the program has already terminated with an error.

### Notes

- Do not run online msDebug while exception dumping is enabled.
- Use the `kernel.o` that exactly matches the failed execution whenever possible.
- Identical Python source does not imply an identical binary. Block size, autotuning, compiler versions, and backend options can all change the binary.
- Hardware may execute several instructions after the fault before reporting it. Some memory and register values in the core dump may therefore differ from those at the initial fault.
- The exception PC is generally more reliable than ordinary data. Start with the PC, stop reason, and disassembly.
- The official documentation describes `bt` primarily for core dumps, with accuracy guaranteed only for certain hardware-exception stop reasons.

### Examples

#### Generating an Operator Core File

After exiting online msDebug, set the following in a normal terminal:

```bash
export ASCEND_DUMP_SCENE=aic_err_detail_dump
export ASCEND_DUMP_PATH=/data/ascend_core_dump
mkdir -p "$ASCEND_DUMP_PATH"
python triton_msdebug_demo.py
```

A `.core` file is generated only when a corresponding device exception occurs.

#### Loading a Core File and `kernel.o`

```bash
msdebug --core /data/ascend_core_dump/<error>.core \
  /data/triton_msdebug_dump/<hash>/kernel.o
```

If the binary is not available yet, load only the core file:

```bash
msdebug --core /data/ascend_core_dump/<error>.core
```

Without a matching `kernel.o`, source-line, symbol, and call-stack resolution is limited.

#### Viewing the Exception Summary

```text
(msdebug) ascend info summary
(msdebug) ascend info devices
(msdebug) ascend info cores
(msdebug) ascend info blocks -d
```

#### Switching to an Affected Core

```text
(msdebug) ascend aiv <id>
(msdebug) ascend aic <id>
```

Select a core whose `stop reason` is an exception such as `MTE_ERROR`, `VEC_ERROR`, or `CUBE_ERROR`, then inspect it:

```text
(msdebug) bt
(msdebug) frame select 0
(msdebug) register read -a
```

#### Locating Instructions from the PC

```text
(msdebug) re r $PC
(msdebug) image list -o -f
(msdebug) image lookup -a <pc-address>
(msdebug) disassemble -s <start-address> -e <end-address>
```

#### Reading Memory from the Core Dump

Use the memory segments listed by `ascend info summary` and the current core context:

```text
(msdebug) x -m GM -f uint8_t[] <address> -s 128 -c 1
(msdebug) x -m STACK -f uint8_t[] <address> -s 128 -c 1
(msdebug) x -m DCACHE -f uint8_t[] <address> -s 128 -c 1
```

Check `help memory read` for the memory types supported in core dumps.

## Troubleshooting

### msDebug Initialization Fails with `0x20102`

Typical message:

```text
msdebug failed to initialize, please run "echo 1 > /proc/debug_switch"
```

Check:

```bash
echo 1 > /proc/debug_switch
cat /proc/debug_switch
ls -l /dev/drv_debug
```

If you cannot change the setting inside a container, configure it on the host.

### A Breakpoint Is Hit, but `var` Shows No Variables

Breakpoints use `.debug_line`; variables use `.debug_info` and `.debug_loc*`. Check the following in order:

```bash
echo "$LLVM_EXTRACT_DI_LOCAL_VARIABLES"
echo "$TRITON_DISABLE_LINE_INFO"
llvm-dwarfdump --verify "$KERNEL_O"
llvm-dwarfdump --debug-info "$KERNEL_O" | less
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
```

If a variable has no location, or the current PC falls outside its location list, msDebug cannot print it.

### Can msDebug Set a Breakpoint in `NormalizeDebugLineLocations.cpp`?

No. This is a host-side MLIR pass. Use compiler logs, IR before and after the pass, and GDB/LLDB to debug `triton-opt`. msDebug debugs the `kernel.o` already loaded onto the NPU.

### UB Data Does Not Match Expectations

Check in order:

1. The core in focus shown by `ascend info cores`.
2. Whether execution stopped after `tl.load` completed and before memory was reused.
3. Whether the data type matches `-f`.
4. Whether `-s` is being interpreted as bytes.
5. Whether the mask makes some lanes invalid.
6. Whether the block size matches the actual vector length.
7. Whether a different autotuned kernel is being debugged.
8. Whether the address is a UB offset or an absolute GM address.

### Does `LD_PRELOAD detected` Require Action?

This warning alone is usually not the cause of a failure. If msDebug initializes, hits breakpoints, and reads device information, you can keep the environment as it is. Check the libraries referenced by `LD_PRELOAD` if library conflicts, symbol-loading errors, or process-initialization failures occur.

## Appendix: Inspecting Debug Information in `kernel.o`

### Inspecting Sections

```bash
readelf -S "$KERNEL_O" | grep -E 'debug|symtab|strtab'
```

### Validating DWARF

```bash
llvm-dwarfdump --verify "$KERNEL_O"
```

### Inspecting the Line Table

```bash
llvm-dwarfdump --debug-line "$KERNEL_O" | less
```

### Inspecting Compilation Units and Variables

```bash
llvm-dwarfdump --debug-info "$KERNEL_O" | less
```

### Inspecting Variable Locations

```bash
llvm-dwarfdump --debug-loc "$KERNEL_O" | less
llvm-dwarfdump --debug-loclists "$KERNEL_O" | less
```

### Saving Analysis Results

```bash
llvm-dwarfdump --verify "$KERNEL_O" > dwarf-verify.txt 2>&1
llvm-dwarfdump --debug-info "$KERNEL_O" > dwarf-info.txt
llvm-dwarfdump --debug-line "$KERNEL_O" > dwarf-line.txt
llvm-dwarfdump --debug-loc "$KERNEL_O" > dwarf-locations.txt
```

When reporting a Triton-Ascend debugging issue, include a minimal Python example, version information, environment variables, the exact matching `kernel.o`, the DWARF output above, msDebug session logs, the target device/core/block, and the actual stop PC.

## References

- [msDebug User Guide](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)
- [msDebug Online User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msdebug_user_guide.md)
- [CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html#single-stepping)
- [DWARF Location Lists](https://dwarfstd.org/doc/040408.1.html)
- [LLVM Source-Level Debugging](https://llvm.org/docs/SourceLevelDebugging.html)
