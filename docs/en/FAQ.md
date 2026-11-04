# Triton-Ascend FAQ

## 1. Installation and Environment Configuration

**Q: How can I correctly install Triton-Ascend? Is it possible to install it directly using pip?**

A: You can directly use pip to install it.

```bash
pip install triton-ascend
```

**Q: Can community Triton and Triton-Ascend coexist?**

A: For Triton-Ascend 3.2.0 and earlier versions, you need to uninstall the community Triton first before installing Triton-Ascend.
 For Triton-Ascend 3.2.1 and later versions, Triton-Ascend declares Triton as an installation dependency to mitigate the installation overwriting issue. When installing
Triton-Ascend, the community Triton is installed first,and the Triton-Ascend overwrites the directory with the same name.
This prevents the installation of triton from overwriting Triton-Ascend when other software packages that depend on Triton are installed.
The reason why x86 and arm use different versions of the community Triton installation package is that the community provides the arm installation package only from version 3.2.1 onwards.
Specifically,x86 depends on triton==3.2.0,and arm depends on triton==3.5.0.

- Note: If you install a third-party software or triton itself that depends on Triton after installing Triton-Ascend,the installed Triton-Ascend directory will be overwritten.
In this case, you also need to uninstall the community Triton and Triton-Ascend first before installing Triton-Ascend.

```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: Can Triton-Ascend be used on non-Ascend hardware (such as CUDA AMD)?**

A: No. Triton-Ascend can be used only in the Ascend NPU hardware environment.

## 2. Accuracy and Numerical Consistency Issues

**Q: How can I troubleshoot the inconsistency between the NPU running result and the PyTorch/CPU/GPU reference result?**

A: For details, see [07_accuracy_comparison_example.md](../en/examples/07_accuracy_comparison_example.md).
For details about the debugging method, see [Debugging in Interpreter Mode](./debug_guide/debugging.md#debug-interpreter-mode).

**Q: What are the integer division / remainder semantics on Ascend when the divisor is zero?**

A: On Ascend, integer remainder/division by a zero divisor is deterministic and compiler-defined (verified with CANN 9.2.0 on Ascend 910B; the guard is inserted by the BiShengIR HFusion `NormalizeModOp` pass, which rewrites integer `mod` into `(y == 0) ? -1 : x % y`):

| Operation | Result on zero-divisor lanes |
|---|---|
| `x % 0`, signed (i8/i16/i32/i64) | `-1` — the all-ones value of the result type |
| `x % 0`, unsigned (e.g. uint8) | `2**n - 1` (e.g. `255`) — the same all-ones bit pattern. Unsigned remainder on widths above 8 bits currently fails to compile on Ascend 910B |
| `x // 0`, int32 (loaded, splat-zero or literal-zero divisor) and int64 (loaded zero divisor) | Saturates: `x > 0 -> MAX`, `x < 0 -> MIN`, `x == 0 -> 0`. `// 0` on i8/i16 currently returns `{-1, 0}` and is excluded from this promise |

The remainder guarantee holds for the tested divisor forms (the loaded tensor form for every observed width; the in-kernel splat zero and literal `0` forms pinned for int32/int64). On non-zero lanes `%` follows C-truncation (`np.fmod`) remainder semantics. These values are compiler-defined guarantees, not hardware accidents; they deliberately differ from eager `torch.remainder`/`torch.fmod` and from CUDA, where the result is architecturally undefined. Interpreter mode (`TRITON_INTERPRET=1`) matches the compiled values for the remainder promise; for `x // 0` the interpreter saturates for every width and divisor form, while the compiled kernels currently return the dividend for int64 splat/literal zero divisors and `{-1, 0}` for i8/i16 — exactly the configurations excluded above. This contract is verified on Ascend 910B with CANN 9.2.0 using the default SIMD (HFusion) pipeline; SIMT mode and other SoCs are outside the verified baseline. If your kernel needs other semantics, guard explicitly, for example `tl.where(y == 0, fallback, x % y)`. The behaviour is pinned by `third_party/ascend/unittest/pytest_ut/test_int_div_zero.py`, whose compiled-device cases skip on non-910B archs.

## 3. Error Code and Exception Handling

**Q: Why is the error message "MLIRCompilationError" displayed during kernel compilation? How can I locate the failed pass?**

A: For details, see [Compilation Error Debugging](./debug_guide/debugging.md#debug-compilation-error).

## 4. Debugging and Logging

**Q: How can I enable detailed log output? Where is the output of TRITON_DEBUG=1?**

A: You can use **TRITON_DEBUG=1** to obtain detailed dump files for debugging. For details, see [Dump Files](./debug_guide/debugging.md#debug-dump-files).

**Q: Can I print the intermediate tensor value in the kernel? Is tl.device_print available?**

A: You can use tl.device_print to print the tensor in the kernel. For details, see [Debugging by Printing](./debug_guide/debugging.md#debug-printing).

## 5. Development and Contributions

**Q: How can I build and test Triton-Ascend locally?**

A: For details about the local build and test methods, see [Installing Triton-Ascend Using the Source Code](./installation_guide.md#install-from-source).

**Q: What CI checks are required for submitting a PR?**

A: The CI checks for a PR include: coding security and specifications check, open-source code check, malicious code check, compilation and building, and developer testing.

## 6. Performance Optimization

**Q: Is there any performance analysis tool (profiler) available?**

A: There is an integrated performance analysis tool (profiler). For details, see [Operator Performance Optimization Methods](./debug_guide/profiling.md).

## 7. Common UB Overflow Issues

**Q: How to resolve "UB Overflow" errors during compilation?**

A: UB Overflow is a common issue in Triton-Ascend development. For details, see [UB Overflow Troubleshooting Guide](./debug_guide/ub_overflow.md) to troubleshoot the issue. If you're unsure how to reduce tiling to lower UB usage, you can use Autotune to automatically select the optimal configuration. For details, see [Triton-Ascend Autotune Guide](./autotune_guide.md).

When migrating operators from Ascend 950PR&950DT products to Atlas A2 products/Atlas A3 products, UB size differences may cause UB Overflow. If manual troubleshooting doesn't resolve the issue, Autotune can also be used to automatically select the optimal configuration.

## 8. Triton Usage Constraints

**Q: What are the usage constraints for pointer parameters in Triton kernels?**

A: The Triton-Ascend compiler assumes at compile time that all externally input pointer parameters essentially point to different memory regions and cannot identify pointer alias scenarios. When multiple pointer parameters actually point to the same memory at runtime but this fact cannot be known at compile time, it may result in optimization failures or abnormal runtime results. For example:

```Python
@triton.jit
def func(ptr0, ptr1):
    # load from ptr0 and do something
    # store to ptr0
    # load from ptr1 and do something
    # store to ptr1

in_out_tensor = torch.randn(shape)
func[grid](in_out_tensor, in_out_tensor)
```

In the above code, `ptr0` and `ptr1` actually point to the same memory (i.e., the same `in_out_tensor`), but the compiler cannot identify this pointer alias relationship. Therefore, passing the same tensor as multiple pointer parameters is not supported, and the corresponding kernel will not be able to enable related optimizations.
