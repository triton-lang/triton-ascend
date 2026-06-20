#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

VALIDATOR = "validate_add_rms_norm.py"


def find_triton_python(op_dir: Path) -> Path:
    for parent in [op_dir, *op_dir.parents]:
        candidate = parent / "python" / "triton"
        if candidate.is_dir():
            return parent / "python"
    raise SystemExit("missing local triton package; expected python/triton under the triton-ascend checkout")


def ensure_triton_available(op_dir: Path) -> None:

    def require_compiled_triton() -> None:
        import triton  # noqa: F401
        import triton._C.libtriton  # noqa: F401

    try:
        require_compiled_triton()
        return
    except ModuleNotFoundError:
        sys.path.insert(0, str(find_triton_python(op_dir)))
    try:
        require_compiled_triton()
    except Exception as exc:
        raise SystemExit("Triton-Ascend must be importable with the compiled triton._C.libtriton extension; "
                         "run this script with the same Python environment used by OpForge/CANN-Bench.") from exc


def main() -> None:
    op_dir = Path(__file__).resolve().parent
    validator = op_dir / VALIDATOR
    if not validator.is_file():
        raise SystemExit(f"missing validator: {validator}")
    ensure_triton_available(op_dir)
    sys.path.insert(0, str(op_dir))
    sys.argv[0] = str(validator)
    runpy.run_path(str(validator), run_name="__main__")


if __name__ == "__main__":
    main()
