# Triton Ascend Latest Updates

This page describes the latest additions and changes to the official Triton Ascend documentation, helping you stay up to date with documentation changes.

Release plans, maintenance policies, and version compatibility information are out of scope for this page; see the [Release Notes](./release_note.md).

> **Security Notice**: If you discover a security issue during use, report it through the channels provided in the [Security Note](https://triton-ascend.readthedocs.io/en/latest/community/SECURITYNOTE.html) and do not disclose it in public channels.

## October 2026

| No. | Update | Description | Related Document |
| --- | --- | --- | --- |
| 1 | Added the msDebug User Guide and the Warmup User Guide | Added a user guide for the msDebug debugging tool and a guide describing the pre-compilation behavior of `kernel.warmup`, both integrated into the documentation site navigation. | [msDebug User Guide](./debug_guide/triton_msdebug_user_guide.md) |
| 2 | Updated the Autotune Guide | Added new sections such as "What Triton Ascend Adds Beyond Community Autotune" (parallel compilation of multiple candidate configs, profiler-based kernel performance collection) and "Scope and Behavior of Automatic Tiling Generation". | [Autotune Guide](./autotune_guide.md) |
| 3 | Updated the Installation Guide | Adapted to the 3.6.0 software stack (CANN 9.2.0, torch_npu 2.10.0) and added a `--no-build-isolation` tip for faster builds. | [Installation Guide](./installation_guide.md) |
| 4 | Updated the libdevice API documentation | Unified the precision assertions in libdevice examples to 1e-4 and removed the `TRITON_ENABLE_LIBDEVICE_SIMT` boilerplate from the examples to improve consistency. | [libdevice API](https://triton-ascend.readthedocs.io/en/latest/python-api/triton.language.extra.cann.libdevice.html) |
| 5 | Updated the Environment Variables and Compiler Options reference | Added a `triton.knobs` interface section and updated the mappings between legacy and new compiler options as well as the migration notes. | [Environment Variables and Compiler Options](./environment_variable_and_compiler_options_reference.md) |
