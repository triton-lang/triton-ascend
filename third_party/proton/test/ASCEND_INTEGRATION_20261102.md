# MSPTI and instrumentation integration validation

## Candidate

- PR #2070: c7e1dc0af961a524cd8b645d6d8e4dec52a1b398.
- PR #2628: 63102a117e4068f6ae66c695d82dfbc4900f76cf.
- Upstream main-dev included by #2628: 2a00c6052a110b6f52a2912d1eb8641ba135d9c3.
- Branch: codex/proton-mspti-instrumentation-20261102.

The merge preserves both PR histories. Device information and naming follow
#2070: DeviceType::NPU and Driver/GPU/CannApi. Both profilers use the same CANN
device queries; unavailable bus width remains zero. The duplicate AscendApi
implementation and its hard-coded device specifications are removed.

NPU defaults to backend="mspti"; instrumentation requires an explicit backend.
AscendRuntime remains enabled alongside MSPTI. CUDA/HIP headers and runtime
implementations are conditional, and setup_patch.py supplies JSON, pybind11,
CANN and MSPTI dependencies without requesting CUDA/CUPTI downloads.

## Environment

Validation ran on Ascend950PR_9579, physical device 3, CANN 9.1.0 and Python
3.11.15, using an isolated package under
/home/j00957045/proton-combined-20261102/.validation. The existing system Triton,
CANN libraries and NPU-IR runtime were not overwritten.

The TA C++ build uses the upstream IR gitlink
4eb12142e16fc8c7ed33c3b568658b773ae0d1c9 and LLVM package
/data/shared/packages/llvm-f6ded0be-3f5016d1-ubuntu24.04-x64.
Device compilation uses the requested external compiler:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/home/j00957045/ascendnpu-ir-pr4196/runtime/bin:$PATH
export LD_PRELOAD=/usr/local/Ascend/cann-9.1.0/lib64/libmspti.so
export ASCEND_RT_VISIBLE_DEVICES=3
```

PR4196 runtime is a source-built f591fb0c compiler with prebuilt 1025 device
templates, as documented in its existing BUILD_PR4196.md; it is not a fully
source-built template package.

SHA-256:

- bishengir-compile: 5bf89d67d1def1151ed21a56ee718e0af4b5ddcad67840d103ae161bfd0ed822.
- meta_op.aiv.c310.O2.bc: 1299ed10e35ba817ed8107df5385d5ebab4d7264a2683f75bca69dc1a3545d73.
- CANN libmspti.so: d71b1065879d4420ad0e9d9009ac3405d3ed57e6faed5cfae4d44aea32834d54.

## Results

| Check | Result |
| --- | --- |
| Joint TA/Proton build with TRITON_OFFLINE_BUILD=1 | PASS |
| CUDA/HIP profiler runtime compile definitions | Absent; CANN and MSPTI enabled |
| Wheel build and isolated installation | PASS |
| Wheel CUDA/CUPTI headers and libraries | None found |
| MSPTI smoke | PASS: 5 launches, 5237 ns aggregate device duration |
| MSPTI lifecycle | 7 passed |
| Default MSPTI, metadata and deactivate/activate, with/without Triton hook | 2 passed |
| Proton Viewer, including installed CLI entry point | 22 passed |
| Native launcher plan and C export ABI | 296 passed |
| Instrumentation off/on/off, 256 and 4096 elements | 2 failed during device linking |

Lifecycle coverage includes state without an explicit hook, multiple sessions,
restart after finalize, multiple streams, multiple threads and Chrome Trace.
Instrumentation tests successfully execute the initial uninstrumented kernel
with profile_scratch_size=0, but fail on the instrumented launch. Consequently,
event counts, durations and the final off transition are not validated on A5.
The A3-specific test in #2628 is not counted as a pass on this A5 machine.

Initial validation harness issues were corrected: Viewer needed the wheel's
console scripts, and launcher tests needed the xdist worker_id fixture. A
separate wheel-installed run passed all 296 tests in 65.66 seconds and is
recorded in launcher-wheel-host.log. This host-side ABI/grid suite runs without
LD_PRELOAD; all MSPTI hardware tests use the CANN library preload. An earlier
wheel-installed launcher run with the preload exceeded its six-minute timeout
while executing native probe subprocesses; its partial log is retained as
launcher-wheel.log.

Formatting was checked with YAPF 0.43.0, clang-format 19.1.6 and Ruff 0.9.1.
The follow-up commit adds tests and this record, and normalizes inherited CRLF
files. Tracked product-source differences from the tested merge commit vanish
with git diff -w --ignore-space-at-eol.

## Instrumentation blocker

The first error is:

```text
ld.lld: error: undefined symbol: _mlir_ciface_proton_circular_store
>>> referenced by LLVMDialectModule
>>> kernel.o:(proton_circular_store)
```

Later diagnostics include "hivm.hir.custom op Missing implementation function
name"; these are not the first failure. The saved TTAdapter contains
__builtin_proton_get_sys_cnt and __builtin_proton_circular_store.
PR4196 bishengir-opt accepts this IR. Compiling the saved IR directly with
bishengir-compile --target=Ascend950 reproduces the linker failure without
Python, the launcher or MSPTI.

The generated object has an undefined reference to
_mlir_ciface_proton_circular_store. The c310 O2 template defines
proton_circular_store and proton_get_sys_cnt but does not define that required
C-interface symbol. This demonstrates a compiler/template call-contract gap
in the tested stack. A compatible implementation/export must be supplied and
the hardware trace assertions rerun before claiming A5 instrumentation works.
No symbol alias, stub or profiling bypass was added to make the tests pass.

## Reproduction

After installing the candidate wheel and configuring the environment above:

```bash
python3 third_party/proton/test/test_ascend_proton_smoke.py
python3 -m pytest -v third_party/proton/test/test_ascend_proton_lifecycle.py
python3 -m pytest -v third_party/proton/test/test_ascend_proton_backends.py
python3 -m pytest -q third_party/proton/test/test_viewer.py
python3 -m pytest -q third_party/ascend/unittest/pytest_ut/test_launcher_plan.py third_party/ascend/unittest/pytest_ut/test_launcher_export_api.py
proton-viewer -m time/us,count third_party/proton/test/ascend_proton_smoke.hatchet
```

Server evidence is under .validation/logs and .validation/repro in the candidate
directory. Build/test/reproducer scripts are /home/j00957045/proton-combined-build.sh,
/home/j00957045/proton-combined-test.sh and
/home/j00957045/proton-instrumentation-reproduce.sh.
The standalone host regression entry is
/home/j00957045/proton-launcher-host-test.sh.
