#!/usr/bin/env python3
"""Patch npuir templates so they build with publicly available bisheng
compilers (CANN 9.0/9.1, clang 15.0.5).

npuir master's device-print templates rely on three things no public compiler
provides:
  1. CCE_PRINT_CC / CCE_DEBUG_NAME marker macros (newer compiler only) —
     defined transparently here, restoring the pre-refactor behavior (the
     previous npuir pin declared these functions without the markers).
  2. cce::printf does not accept `unsigned long` (uint64_t on LP64) in its
     Support<> whitelist — the uint64_t print registrations are removed.
  3. The SIMT Debug variant exports the same _mlir_ciface_print_* symbols as
     the regular variant; llvm-link rejects both in one group. The SIMT
     subdirectory is skipped.

Remove this script once a compiler supporting these features is released.

Usage: patch-npuir-templates.py <npuir-source-root>
"""
import sys
from pathlib import Path


def patch_debug_utils(root: Path) -> None:
    p = root / "bishengir/lib/Template/include/Debug/DebugUtils.h"
    s = p.read_text()
    shim = (
        "#ifndef CCE_PRINT_CC\n#define CCE_PRINT_CC\n#endif\n"
        "#ifndef CCE_DEBUG_NAME\n#define CCE_DEBUG_NAME(x) x\n#endif\n"
    )
    assert s.count('#include "Utils.h"') == 1, "unexpected DebugUtils.h content"
    p.write_text(s.replace('#include "Utils.h"\n', '#include "Utils.h"\n' + shim))
    print(f"patched {p}")


def patch_debug_cpp(root: Path) -> None:
    p = root / "bishengir/lib/Template/lib/Debug/Debug.cpp"
    s = p.read_text()
    for line in (
        "REGISTER_PRINT_SCALAR(uint64_t, gm)",
        "REGISTER_PRINT_1TO8D_TENSOR(uint64_t, gm)",
        "REGISTER_PRINT_SCALAR(uint64_t, ubuf)",
        "REGISTER_PRINT_1TO8D_TENSOR(uint64_t, ubuf)",
    ):
        assert line in s, f"missing {line} in Debug.cpp"
        s = s.replace(line + "\n", "")
    p.write_text(s)
    print(f"patched {p}")


def patch_regbase_debug_cmake(root: Path) -> None:
    p = root / "bishengir/lib/Template/lib/RegBase/Debug/CMakeLists.txt"
    s = p.read_text()
    assert "add_subdirectory(SIMT)" in s, "unexpected RegBase/Debug/CMakeLists.txt"
    p.write_text(s.replace("add_subdirectory(SIMT)\n", ""))
    print(f"patched {p}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    patch_debug_utils(root)
    patch_debug_cpp(root)
    patch_regbase_debug_cmake(root)


if __name__ == "__main__":
    main()
