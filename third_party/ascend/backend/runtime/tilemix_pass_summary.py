from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

_BEFORE_HEADER = re.compile(
    r"^\s*//\s*-+// IR Dump Before TileCubeVectorLoop \(tile-cube-vector-loop\) //-+\s*//\s*$",
    re.MULTILINE,
)
_AFTER_HEADER = re.compile(
    r"^\s*//\s*-+// IR Dump After TileCubeVectorLoop \(tile-cube-vector-loop\) //-+\s*//\s*$",
    re.MULTILINE,
)
_WARNING = re.compile(
    r"Ignoring candidate\s+(cube|vector)\s+loop trip count because it's\s+([^\r\n]+)",
    re.IGNORECASE,
)

_SUMMARY_KEYS = (
    "tile_mix_summary_source",
    "tile_mix_summary_valid",
    "tile_mix_cube_applied",
    "tile_mix_vector_applied",
    "tile_mix_cube_segments",
    "tile_mix_vector_segments",
    "tile_mix_cube_skip_reason",
    "tile_mix_vector_skip_reason",
    "tile_mix_sync_ops_before",
    "tile_mix_sync_ops_after",
)


def should_capture_tilemix_pass_summary(cube_loop: Optional[int], vector_loop: Optional[int],
                                        env: Mapping[str, str]) -> bool:
    """Return whether the compiler should capture TileCubeVectorLoop before/after IR."""
    # Primary costmodel pruning must stop at TTIR. Real HIVM evidence is an
    # explicit second-stage validation mode, so capture is disabled by default.
    mode = env.get("TRITON_CAPTURE_TILE_MIX_PASS_SUMMARY", "0").strip().lower()
    if mode in {"0", "off", "false", "no"}:
        return False
    return any(isinstance(value, int) and value > 1 for value in (cube_loop, vector_loop))


def add_tilemix_pass_dump_options(options, cube_loop, vector_loop, env=None) -> bool:
    """Append the exact pass probes once and return whether capture is enabled."""
    import os

    if not should_capture_tilemix_pass_summary(cube_loop, vector_loop, env or os.environ):
        return False
    before = "--mlir-print-ir-before=tile-cube-vector-loop"
    after = "--mlir-print-ir-after=tile-cube-vector-loop"
    if before not in options:
        options.append(before)
    if after not in options:
        options.append(after)
    return True


def _normalize_ir(text: str) -> str:
    # SSA numbering and locations are deliberately retained. A pass that only
    # renumbers values still changed its output; treating that as unknown is
    # safer than claiming that TileMix did not run.
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _sync_op_count(text: str) -> int:
    return sum(
        text.count(token) for token in (
            "hivm.hir.sync_block_set",
            "hivm.hir.sync_block_wait",
            "hivm.hir.set_flag",
            "hivm.hir.wait_flag",
            "hivm.hir.pipe_barrier",
        ))


def _requested(value: Optional[int]) -> int:
    try:
        return max(1, int(value)) if value is not None else 1
    except (TypeError, ValueError):
        return 1


def parse_tilemix_pass_summary(compiler_output: str, *, cube_loop: Optional[int],
                               vector_loop: Optional[int]) -> Dict[str, object]:
    """Build a fail-closed summary from the real TileCubeVectorLoop invocation.

    The proprietary pass currently exposes no callback metadata. Its supported
    diagnostic surface is the pass-before/pass-after IR and pass warnings. We
    therefore only report a side as applied when the pass produced a structural
    change and did not explicitly reject that side. Ambiguous evidence remains
    invalid and downstream modeling must use the Off score.
    """
    cube_requested = _requested(cube_loop)
    vector_requested = _requested(vector_loop)
    result: Dict[str, object] = {
        "schema_version": 1,
        "source": "tile_cube_vector_loop_ir_diff",
        "valid": False,
        "changed": False,
        "cube_requested_segments": cube_requested,
        "vector_requested_segments": vector_requested,
        "cube_applied": False,
        "vector_applied": False,
        "cube_segments": 1,
        "vector_segments": 1,
        "cube_skip_reason": "missing_pass_dump",
        "vector_skip_reason": "missing_pass_dump",
        "sync_ops_before": 0,
        "sync_ops_after": 0,
    }

    before_match = _BEFORE_HEADER.search(compiler_output or "")
    after_match = _AFTER_HEADER.search(compiler_output or "")
    if not before_match or not after_match or before_match.end() >= after_match.start():
        return result

    before = compiler_output[before_match.end():after_match.start()]
    # Compiler warnings and linker diagnostics follow the after dump. They are
    # not IR and must not make an unchanged pass look changed.
    after_tail = compiler_output[after_match.end():]
    diagnostic_positions = [
        pos for pos in (after_tail.find("\nloc("), after_tail.find("\nwarning:"), after_tail.find("\nld.lld:"))
        if pos >= 0
    ]
    after = after_tail[:min(diagnostic_positions)] if diagnostic_positions else after_tail
    changed = _normalize_ir(before) != _normalize_ir(after)

    warnings = {"cube": [], "vector": []}
    for match in _WARNING.finditer(compiler_output):
        side = match.group(1).lower()
        reason = re.sub(r"[^a-z0-9]+", "_", match.group(2).strip().lower()).strip("_")
        warnings[side].append(f"pass_rejected_{reason or 'unknown'}")

    def side_state(side: str, requested: int):
        if requested <= 1:
            return False, 1, "requested_loop_le_one"
        if warnings[side]:
            return False, 1, warnings[side][0]
        if not changed:
            return False, 1, "pass_no_structural_change"
        core_marker = f"#hivm.tcore_type<{side.upper()}>"
        if core_marker not in after:
            return False, 1, f"no_{side}_loop_after_pass"
        return True, requested, "none"

    cube_applied, cube_segments, cube_reason = side_state("cube", cube_requested)
    vector_applied, vector_segments, vector_reason = side_state("vector", vector_requested)
    result.update({
        "valid": True,
        "changed": changed,
        "cube_applied": cube_applied,
        "vector_applied": vector_applied,
        "cube_segments": cube_segments,
        "vector_segments": vector_segments,
        "cube_skip_reason": cube_reason,
        "vector_skip_reason": vector_reason,
        "sync_ops_before": _sync_op_count(before),
        "sync_ops_after": _sync_op_count(after),
    })
    return result


def summary_to_compile_params(summary: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """Flatten compiler metadata into vTriton's comma-separated option contract."""
    if not isinstance(summary, Mapping):
        return {}
    values = {
        "tile_mix_summary_source": summary.get("source", "unknown"),
        "tile_mix_summary_valid": int(bool(summary.get("valid", False))),
        "tile_mix_cube_applied": int(bool(summary.get("cube_applied", False))),
        "tile_mix_vector_applied": int(bool(summary.get("vector_applied", False))),
        "tile_mix_cube_segments": summary.get("cube_segments", 1),
        "tile_mix_vector_segments": summary.get("vector_segments", 1),
        "tile_mix_cube_skip_reason": summary.get("cube_skip_reason", "unknown"),
        "tile_mix_vector_skip_reason": summary.get("vector_skip_reason", "unknown"),
        "tile_mix_sync_ops_before": summary.get("sync_ops_before", 0),
        "tile_mix_sync_ops_after": summary.get("sync_ops_after", 0),
    }
    return {key: values[key] for key in _SUMMARY_KEYS}
