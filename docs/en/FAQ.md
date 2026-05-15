# Triton-Ascend FAQ

## 1. Installation and Environment Configuration

**Q: How can I correctly install Triton-Ascend? Is it possible to install it directly using pip?**

A: You can directly use pip to install it.

```Python
pip install triton-ascend
```

**Q: Can community Triton and Triton-Ascend coexist?**

A: For Triton-Ascend 3.2.0 and earlier versions.You need to uninstall the community Triton first before installing Triton-Ascend.<br>
 For Triton-Ascend 3.2.1 and later versions.Triton-Ascend declares Triton as an installation dependency to mitigate the installation overwriting issue.When installing
Triton-Ascend,the community Triton is installed first,and the Triton-Ascend overwrites the directory with the same name.
This prevents the installation of triton from overwriting Triton-Ascend when other software packages that depend on Triton are installed.
The reason why x86 and arm use different versions of the community Triton installation package is that the community provides the arm installation package only form version 3.2.1 onwards.
Specifically,x86 depends on triton==3.2.0,and arm depends on triton==3.5.0.

- Note: If you install a third-party software or triton itself that depends on Triton after installing Triton-Ascend,the installed Triton-Ascend directory will be overwritten.
In this case, you also need to uninstall the community Triton and Triton-Ascend first before installing Triton-Ascend.

```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: Can Triton-Ascend be used on non-Ascend hardware (such as CUDA AMD)?**

A: No. Triton-Ascend can be used only in the Ascend NPU hardware environment.

## 2. Accuracy and Numerical Consistency Issues

**Q: How can I troubleshoot the inconsistency between the NPU running result and the PyTorch/CPU/GPU reference result?**

A: For details, see [07_accuracy_comparison_example.md](../en/examples/07_accuracy_comparison_example.md).
For details about the debugging method, see [Debugging in Interpreter Mode](./debug_guide/debugging.md#4-interpreter-mode).

## 3. Error Code and Exception Handling

**Q: Why is the error message "MLIRCompilationError" displayed during kernel compilation? How can I locate the failed pass?**

A: For details, see [Compilation Error Debugging](./debug_guide/debugging.md#52-compilation-error-debugging).

## 4. Debugging and Logging

**Q: How can I enable detailed log output? Where is the output of TRITON_DEBUG=1?**

A: You can use **TRITON_DEBUG=1** to obtain detailed dump files for debugging. For details, see [Dump Files](./debug_guide/debugging.md#32-dump-files).

**Q: Can I print the intermediate tensor value in the kernel? Is tl.device_print available?**

A: You can use tl.device_print to print the tensor in the kernel. For details, see [Debugging by Printing](./debug_guide/debugging.md#51-debugging-by-printing).

## 5. Development and Contributions

**Q: How can I build and test Triton-Ascend locally?**

A: For details about the local build and test methods, see [Installing Triton-Ascend Using the Source Code](./installation_guide.md#installing-triton-ascend-using-the-source-code).

**Q: What CI checks are required for submitting a PR?**

A: The CI checks for a PR include: coding security and specifications check, open-source code check, malicious code check, compilation and building, and developer testing.

## 6. Performance Optimization

**Q: Is there any performance analysis tool (profiler) available?**

A: There is an integrated performance analysis tool (profiler). For details, see [Operator Performance Optimization Methods](./debug_guide/profiling.md).
