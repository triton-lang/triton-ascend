# Triton-Ascend What's New

This page describes the latest launches, additions, and changes to the official Triton-Ascend documentation ([https://triton-ascend.readthedocs.io](https://triton-ascend.readthedocs.io/en/latest/index.html)), organized in reverse chronological order so that users can stay informed of documentation updates.

Release plans, maintenance policies, and supported version information are out of scope for this page; see [Releases](./release_note.md).

> **Security notice**: If you discover a security issue, report it through the channels described in the [Security Notice](./community/SECURITYNOTE.md). Do not disclose it publicly.

## October 2026 (3.6.0)

| No. | Update | Description | Related Documentation |
| --- | --- | --- | --- |
| 1 | New do_bench_npu guide | Added a usage guide for the `do_bench_npu` performance benchmarking tool, covering performance testing methods and result interpretation in NPU scenarios, and added it to the documentation site navigation index. | [do_bench_npu guide](./debug_guide/do_bench_npu_guide.md) |
| 2 | New "User-defined Autotune Feedback" section | The Autotune guide adds a description of the user-defined autotune feedback mechanism, allowing users to accumulate tuning experience and feed it back into the compilation workflow. | [Autotune guide](./autotune_guide.md) |
| 3 | New "Source Build Options" description | The Installation Guide adds a description of the source build options, guiding users to configure build parameters as needed when compiling from source. | [Installation Guide](./installation_guide.md) |
| 4 | Updated the libdevice developer guide | Substantially revised the libdevice developer guide, expanded the operator constraint descriptions, and added examples for operators such as `fast_tanhf`, `float2half_rn`, `half2float`, `max`, `min`, and `nan`. | [libdevice developer guide](./libdevice/libdevice_developer_guide.md) |
| 5 | Updated the Quick Start | Expanded the "Next Steps for Development" index to help users transition smoothly from the Quick Start to advanced documentation such as operator development, operator migration, and debugging and tuning. | [Quick Start](./quick_start.md) |
| 6 | Updated Environment Variables and Compiler Options | Synchronized the compile options cleanup changes and updated the mapping and migration instructions between the old and new compiler options. | [Environment Variables and Compiler Options](./environment_variable_and_compiler_options_reference.md) |
| 7 | Updated the libdevice API documentation | Unified the precision assertions of the libdevice documentation examples to 1e-4, improving the consistency of the examples. | [libdevice API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 8 | Fixed extension API documentation issues | Fixed rendering issues in the extension and bl documentation and added usage examples for the `sub_vec_num` operator. | [Extension Ops API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.extra.cann.extension.html) |
| 9 | Updated community technical meetings | Updated the technical meeting calendar (icalendar) subscription information so community members can subscribe to the meeting schedule. | [Technical meetings](./community/community_technical_meeting.md) |
| 10 | Documentation quality optimization | Fixed documentation typos, formatting, and link issues in batch, and unified the product naming conventions (e.g., Atlas A2/A3). | / |

## July 2026 (3.2.2)

| No. | Update | Description | Related Documentation |
| --- | --- | --- | --- |
| 1 | New libdevice API documentation | Added the auto-generated libdevice (autodoc) documentation, covering usage constraints and runnable examples for more than 60 operators including `acos`, `asin`, `atan`, `fast_dividef`, and `fast_expf`. | [libdevice API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 2 | New triton_launch_kernel API documentation | Added the description and usage examples of the `triton_launch_kernel` extension API in the bl domain. | [Extension Ops API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.extra.cann.extension.html) |
| 3 | New tl.topk operator documentation | Added the API description and examples of the `tl.topk` operator. | [triton.language API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.html) |
| 4 | Added a "Multi-dimensional Tensor Slicing" section to the operator development guide | Added usage and examples for multi-dimensional tensor slicing. | [Programming Guide](./programming_guide/index.md) |
| 5 | Added memory access usage constraints | Added the support status of the `cache_modifier` and `volatile` parameters in `tl.load`/`tl.store`, as well as memory access usage constraints in control flow scenarios. | [triton.language API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.html) |
| 6 | Batch update of operator API documentation | Updated the constraints and examples for math, reduction, memory access, creation, Scan/Sort, and Atomic/Rand/Histogram operators, covering extension operators such as `gather_out_to_ub`, `index_put`, and `scatter_ub_to_out`. | [triton.language API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.html) |
| 7 | Updated the Installation Guide and Quick Start | Optimized the installation flow and quick start content. | [Installation Guide](./installation_guide.md) |
| 8 | Added a device_print example to the Debugging Guide | Added a runtime output example for `device_print`. | [Operator Debugging Guide](./debug_guide/debugging.md) |
| 9 | Added UB Overflow handling to the FAQ | Added troubleshooting and handling instructions for UB Overflow issues and a Triton usage constraint entry. | [FAQ](./FAQ.md) |
| 10 | Added the auto-blockify mechanism to the Migration Guide | Added migration instructions for the auto-blockify automatic blocking mechanism in the architecture difference section. | [Architecture Differences](./migration_guide/architecture_difference.md) |
| 11 | Chinese community documentation is now available | Added Chinese community documentation including the contribution guide, contributor covenant, governance, security notice, technical meetings, and release policy, and added them to the documentation site navigation. | [Contribution Guide](./community/CONTRIBUTING.md) |

## April 2026 (3.2.1)

| No. | Update | Description | Related Documentation |
| --- | --- | --- | --- |
| 1 | Updated the Installation Guide | Updated the LLVM patch link and added instructions to avoid pip installation overwrite issues. | [Installation Guide](./installation_guide.md) |
| 2 | Updated Architecture Design and Core Features | Revised the architecture design documentation content and optimized the wording of the sections. | [Architecture Design and Core Features](./architecture_design_and_core_features.md) |
| 3 | Unified product naming | Standardized CANN type-related naming to a2/a3 in the documentation and fixed inconsistent wording. | / |
| 4 | Updated the Operator Migration Guide | Added migration instructions for migrating operators from GPU to NPU. | [Operator Migration Guide](./migration_guide/index.md) |
| 5 | Updated the Performance Tuning Guide | Updated the profiling-related content for performance tuning. | [Performance Tuning Guide](./debug_guide/profiling.md) |
| 6 | Batch documentation corrections and build optimization | Fixed minor documentation errors and formatting issues and optimized the documentation build workflow (doc-tools, docs CI). | / |
