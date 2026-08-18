# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Post-process the DWARF ``.debug_line`` table of a compiled Ascend kernel so
that ``msdebug`` steps once per user source line.

bishengir emits one ``is_stmt`` row per *static* occurrence of a source line.
Instruction scheduling and inlining duplicate those rows (one statement lowered
to several scattered instruction groups, loop bodies, inlined device-library
frames), so plain ``next`` stepping stops repeatedly on the same line and dives
into compiler-generated code. This module removes the redundant stops while
preserving genuine re-entries (loop back-edges).

It complements the MLIR passes ``CanonicalizeDebugLocationsPass`` and
``DeduplicateDebugNopsPass``: those clean up locations *before* bishengir; this
cleans up the residual scatter bishengir introduces *after* them, at the binary
level.

Mechanism (length-preserving, relocation-safe)
    bishengir encodes the line program as a single relocated
    ``DW_LNE_set_address`` followed by ``DW_LNS_fixed_advance_pc`` per row,
    whose operands are symbol differences patched by ``.rela.debug_line``.
    Regenerating the program would reorder those relocations and corrupt the
    table, so instead each dropped stop has its ``DW_LNS_copy`` opcode (0x01)
    overwritten with ``DW_LNS_set_basic_block`` (0x07) -- both one-byte, zero
    argument. ``set_basic_block`` emits no row, so the stop disappears and its
    address folds into the previous row. No byte shifts, so ``.rela.debug_line``
    is never touched and stays valid.

Demotion rules, applied to ``is_stmt=1`` rows in program order:
    1. Scatter   -- within a sequence keep the first row per ``(file, line)``,
       drop the rest.
    2. Loops     -- keep every occurrence of a ``for``/``while`` header line
       (detected from the source AST), so the back-edge re-stops each iteration.
    3a. Foreign  -- drop rows whose file is not the user source file.
    3b. line==0  -- drop compiler-generated rows.
    3c. Over-len -- drop rows whose line exceeds the source length (safety net
        for binaries built before ``CanonicalizeDebugLocationsPass`` or without
        ``--enable-ms-debug``; normally a no-op).

Rows emitted by a DWARF special opcode (e.g. the line-0 function-entry row)
cannot be edited length-preserving and are left as-is; this is harmless because
execution begins there and ``next`` never returns to it.

Public API
    rewrite_debug_line(artifact, metadata=None, options=None)
        Pipeline entry. No-op unless ``LLVM_EXTRACT_DI_LOCAL_VARIABLES`` is set.
        Accepts the ``bytes`` (or path) produced by the ``npubin`` stage and
        returns the rewritten ``bytes`` (or the same path, patched in place).
        Exception-safe: any failure logs and returns the artifact unchanged.

    rewrite_debug_line_blob(blob, src_path=None) -> (bytes, RewriteResult)
        Pure in-memory transform (no env gate); the unit-testable core.

    rewrite_debug_line_file(path, src_path=None, out_path=None) -> RewriteResult
        File convenience used by the CLI; writes atomically.
"""

from __future__ import annotations

import argparse
import ast
import copy
import io
import logging
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from elftools.elf.elffile import ELFFile
except ImportError:  # surfaced lazily; importing this module must never fail
    ELFFile = None

log = logging.getLogger(__name__)

ENV_FLAG = "LLVM_EXTRACT_DI_LOCAL_VARIABLES"

# ── DWARF line-program opcodes ────────────────────────────────────────────────
DW_LNS_COPY = 0x01
DW_LNS_ADVANCE_PC = 0x02
DW_LNS_ADVANCE_LINE = 0x03
DW_LNS_SET_FILE = 0x04
DW_LNS_SET_COLUMN = 0x05
DW_LNS_NEGATE_STMT = 0x06
DW_LNS_SET_BASIC_BLOCK = 0x07
DW_LNS_CONST_ADD_PC = 0x08
DW_LNS_FIXED_ADVANCE_PC = 0x09
DW_LNS_SET_PROLOGUE_END = 0x0A
DW_LNS_SET_EPILOGUE_BEGIN = 0x0B
DW_LNS_SET_ISA = 0x0C
DW_LNE_END_SEQUENCE = 0x01
DW_LNE_SET_ADDRESS = 0x02
DW_LNE_DEFINE_FILE = 0x03
DW_LNE_SET_DISCRIMINATOR = 0x04


@dataclass
class RewriteResult:
    """Outcome of a rewrite attempt."""
    changed: bool
    demoted: int = 0
    before: List[int] = field(default_factory=list)
    after: List[int] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


# ── LEB128 ────────────────────────────────────────────────────────────────────


def _uleb(data: bytes, off: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, off


def _sleb(data: bytes, off: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if shift < 64 and byte & 0x40:
                result |= -(1 << shift)
            return result, off


# ── ELF access (navigation only; the DWARF resolver is never invoked because it
#    chokes on the .npubin's AArch64 relocations and address_size=0) ───────────


def _read_sections(stream) -> Tuple[Dict[str, dict], bool]:
    if ELFFile is None:
        raise RuntimeError("pyelftools is required (pip install pyelftools)")
    elf = ELFFile(stream)
    little = elf.little_endian
    sections: Dict[str, dict] = {}
    for name in (".debug_line", ".rela.debug_line"):
        sec = elf.get_section_by_name(name)
        if sec is not None:
            sections[name] = {
                "data": sec.data(),
                "offset": sec["sh_offset"],
                "size": sec["sh_size"],
            }
    return sections, little


# ── line-program header ───────────────────────────────────────────────────────


def _parse_header(data: bytes, little: bool) -> dict:
    endian = "<" if little else ">"
    off = 0
    unit_length, = struct.unpack_from(endian + "I", data, off)
    off += 4
    is_dwarf64 = unit_length == 0xFFFFFFFF
    if is_dwarf64:
        unit_length, = struct.unpack_from(endian + "Q", data, off)
        off += 8
    version, = struct.unpack_from(endian + "H", data, off)
    off += 2

    address_size = 8
    if version >= 5:
        address_size = data[off]
        off += 1
        off += 1  # segment selector size

    if is_dwarf64:
        header_length, = struct.unpack_from(endian + "Q", data, off)
        off += 8
    else:
        header_length, = struct.unpack_from(endian + "I", data, off)
        off += 4
    program_start = off + header_length

    min_inst_len = data[off]
    off += 1
    if version >= 4:
        off += 1  # maximum_operations_per_instruction
    default_is_stmt = data[off]
    off += 1
    line_base = struct.unpack_from("b", data, off)[0]
    off += 1
    line_range = data[off]
    off += 1
    opcode_base = data[off]
    off += 1
    standard_opcode_lengths = [0] + [data[off + i] for i in range(opcode_base - 1)]
    off += opcode_base - 1

    include_dirs = [""]
    while data[off] != 0:
        end = data.index(0, off)
        include_dirs.append(data[off:end].decode("utf-8", "replace"))
        off = end + 1
    off += 1

    file_names = [""]
    if version <= 4:
        while data[off] != 0:
            end = data.index(0, off)
            file_names.append(data[off:end].decode("utf-8", "replace"))
            off = end + 1
            _, off = _uleb(data, off)  # dir index
            _, off = _uleb(data, off)  # mtime
            _, off = _uleb(data, off)  # size
        off += 1

    return {
        "version": version,
        "address_size": address_size,
        "min_inst_len": min_inst_len,
        "default_is_stmt": default_is_stmt,
        "line_base": line_base,
        "line_range": line_range,
        "opcode_base": opcode_base,
        "standard_opcode_lengths": standard_opcode_lengths,
        "file_names": file_names,
        "program_start": program_start,
    }


# ── line-number state machine ─────────────────────────────────────────────────


def _simulate(data: bytes, hdr: dict) -> List[List[dict]]:
    """Replay the line program. Each row records ``emit_off`` (byte offset of the
    opcode that produced it) and ``emit_kind`` ('copy' | 'special' | 'end')."""
    ob = hdr["opcode_base"]
    lb = hdr["line_base"]
    lr = hdr["line_range"]
    mil = hdr["min_inst_len"]
    addr_size = hdr["address_size"]
    sol = hdr["standard_opcode_lengths"]
    default_stmt = bool(hdr["default_is_stmt"])

    def fresh() -> dict:
        return dict(addr=0, file=1, line=1, col=0, is_stmt=default_stmt, end_sequence=False)

    sequences: List[List[dict]] = []
    current: List[dict] = []
    state = fresh()
    off = hdr["program_start"]

    while off < len(data):
        op_off = off
        opcode = data[off]
        off += 1

        if opcode == 0:  # extended
            length, off = _uleb(data, off)
            sub = data[off]
            off += 1
            if sub == DW_LNE_END_SEQUENCE:
                row = copy.copy(state)
                row["end_sequence"] = True
                row["emit_off"] = op_off
                row["emit_kind"] = "end"
                current.append(row)
                sequences.append(current)
                current = []
                state = fresh()
            elif sub == DW_LNE_SET_ADDRESS:
                if addr_size == 8:
                    state["addr"], = struct.unpack_from("<Q", data, off)
                    off += 8
                else:
                    state["addr"], = struct.unpack_from("<I", data, off)
                    off += 4
            elif sub == DW_LNE_SET_DISCRIMINATOR:
                _, off = _uleb(data, off)
            elif sub == DW_LNE_DEFINE_FILE:
                end = data.index(0, off)
                off = end + 1
                for _ in range(3):
                    _, off = _uleb(data, off)
            else:
                off += length - 1
        elif opcode < ob:  # standard
            if opcode == DW_LNS_COPY:
                row = copy.copy(state)
                row["emit_off"] = op_off
                row["emit_kind"] = "copy"
                current.append(row)
            elif opcode == DW_LNS_ADVANCE_PC:
                operand, off = _uleb(data, off)
                state["addr"] += operand * mil
            elif opcode == DW_LNS_ADVANCE_LINE:
                operand, off = _sleb(data, off)
                state["line"] += operand
            elif opcode == DW_LNS_SET_FILE:
                state["file"], off = _uleb(data, off)
            elif opcode == DW_LNS_SET_COLUMN:
                state["col"], off = _uleb(data, off)
            elif opcode == DW_LNS_NEGATE_STMT:
                state["is_stmt"] = not state["is_stmt"]
            elif opcode == DW_LNS_SET_BASIC_BLOCK:
                pass
            elif opcode == DW_LNS_CONST_ADD_PC:
                state["addr"] += ((255 - ob) // lr) * mil
            elif opcode == DW_LNS_FIXED_ADVANCE_PC:
                operand, = struct.unpack_from("<H", data, off)
                off += 2
                state["addr"] += operand
            elif opcode in (DW_LNS_SET_PROLOGUE_END, DW_LNS_SET_EPILOGUE_BEGIN):
                pass
            elif opcode == DW_LNS_SET_ISA:
                _, off = _uleb(data, off)
            else:  # unknown standard opcode: skip its ULEB operands
                for _ in range(sol[opcode] if opcode < len(sol) else 0):
                    _, off = _uleb(data, off)
        else:  # special
            adj = opcode - ob
            state["line"] += lb + (adj % lr)
            state["addr"] += (adj // lr) * mil
            row = copy.copy(state)
            row["emit_off"] = op_off
            row["emit_kind"] = "special"
            current.append(row)

    if current:
        sequences.append(current)
    return sequences


# ── source analysis ───────────────────────────────────────────────────────────


def _loop_header_lines(src_path: str) -> set:
    with open(src_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=src_path)
    return {node.lineno for node in ast.walk(tree) if isinstance(node, (ast.For, ast.AsyncFor, ast.While))}


def _source_length(src_path: str) -> int:
    with open(src_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _auto_detect_source(file_names: List[str]) -> Optional[str]:
    """First user (.py, not site-packages) file from the line table that exists
    on disk, resolved relative to the current working directory."""
    for path in file_names:
        if path.endswith(".py") and "/site-packages/" not in path and os.path.isfile(path):
            return path
    return None


def _user_file_indices(file_names: List[str], src_path: str) -> set:
    base = os.path.basename(src_path)
    return {i for i, p in enumerate(file_names) if os.path.basename(p) == base}


# ── demotion planning ─────────────────────────────────────────────────────────


def _plan_demotions(sequences, protected, user_files, src_lines):
    demote: List[dict] = []
    kept: List[int] = []
    counts = {"dup": 0, "foreign": 0, "line0": 0, "over": 0, "special_skip": 0}
    for sequence in sequences:
        seen = set()
        for row in sequence:
            if row["end_sequence"] or not row["is_stmt"]:
                continue
            reason = None
            if user_files is not None and row["file"] not in user_files:
                reason = "foreign"
            elif row["line"] == 0:
                reason = "line0"
            elif src_lines and row["line"] > src_lines:
                reason = "over"
            elif row["line"] in protected:
                reason = None  # keep every occurrence of a loop header
            else:
                key = (row["file"], row["line"])
                if key in seen:
                    reason = "dup"
                else:
                    seen.add(key)
            if reason and row["emit_kind"] != "special":
                demote.append(row)
                counts[reason] += 1
            else:
                if reason:  # wanted but special -> survives unpatched
                    counts["special_skip"] += 1
                kept.append(row["line"])
    return demote, kept, counts


def _surviving_is_stmt_lines(data: bytes, hdr: dict) -> List[int]:
    return [
        row["line"]
        for sequence in _simulate(data, hdr)
        for row in sequence
        if not row["end_sequence"] and row["is_stmt"]
    ]


# ── core transform (in-memory, no env gate) ──────────────────────────────────


def rewrite_debug_line_blob(blob: bytes, src_path: Optional[str] = None) -> Tuple[bytes, RewriteResult]:
    """Rewrite the ``.debug_line`` table inside an ELF ``blob``.

    Returns ``(new_blob, result)``. ``new_blob`` is the original ``blob`` when
    no change is made or verification fails (the input is never corrupted).
    """
    sections, little = _read_sections(io.BytesIO(blob))
    if ".debug_line" not in sections:
        return blob, RewriteResult(False, reason="no .debug_line section")

    debug_line = sections[".debug_line"]["data"]
    hdr = _parse_header(debug_line, little)
    sequences = _simulate(debug_line, hdr)

    src = src_path or _auto_detect_source(hdr["file_names"])
    protected = _loop_header_lines(src) if src and os.path.isfile(src) else set()
    user_files = _user_file_indices(hdr["file_names"], src) if src else None
    # If the source matches no file in the line table the foreign-file rule would
    # demote every row; that is almost certainly a wrong source path, so disable
    # the rule rather than wipe all stops.
    if user_files is not None and not user_files:
        log.warning("debug-line: source %s matches no file in the line table %s; "
                    "skipping foreign-file rule", src, hdr["file_names"][1:])
        user_files = None
    src_lines = _source_length(src) if src and os.path.isfile(src) else None

    before = [r["line"] for s in sequences for r in s if r["is_stmt"] and not r["end_sequence"]]
    demote, kept, counts = _plan_demotions(sequences, protected, user_files, src_lines)

    if not demote:
        return blob, RewriteResult(False, before=before, after=kept, counts=counts, reason="nothing to demote")

    base = sections[".debug_line"]["offset"]
    patched = bytearray(blob)
    for row in demote:
        file_off = base + row["emit_off"]
        if patched[file_off] != DW_LNS_COPY:
            return blob, RewriteResult(
                False, before=before, after=kept, counts=counts,
                reason=f"expected Copy(0x01) at .debug_line+{row['emit_off']:#x}, "
                f"found {patched[file_off]:#04x}")
        patched[file_off] = DW_LNS_SET_BASIC_BLOCK

    # Verify against the patched bytes (length-preserving, so offsets are stable).
    verify_sections, _ = _read_sections(io.BytesIO(bytes(patched)))
    survivors = _surviving_is_stmt_lines(verify_sections[".debug_line"]["data"], hdr)
    if survivors != kept:
        return blob, RewriteResult(False, before=before, after=kept, counts=counts,
                                   reason=f"verify mismatch: survivors={survivors} kept={kept}")

    return bytes(patched), RewriteResult(True, demoted=len(demote), before=before, after=kept, counts=counts,
                                         reason="ok")


# ── file convenience (CLI / manual use) ───────────────────────────────────────


def rewrite_debug_line_file(path: str, src_path: Optional[str] = None, out_path: Optional[str] = None) -> RewriteResult:
    """Rewrite ``path``; write to ``out_path`` (or in place when ``None``).
    Writes atomically via a temp file + ``os.replace``."""
    with open(path, "rb") as handle:
        blob = handle.read()
    new_blob, result = rewrite_debug_line_blob(blob, src_path)
    if not result.changed:
        return result

    dest = out_path or path
    dest_dir = os.path.dirname(os.path.abspath(dest)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".debugline_", suffix=".npubin", dir=dest_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(new_blob)
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return result


# ── pipeline entry (env-gated, polymorphic, exception-safe) ───────────────────


def _enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_source(metadata) -> Optional[str]:
    """Best-effort source path from compiler metadata; ``None`` falls back to
    auto-detection from the line table's file_names."""
    if metadata is None:
        return None
    for attr in ("src_path", "source_path", "filename"):
        value = getattr(metadata, attr, None)
        if isinstance(value, str) and os.path.isfile(value):
            return value
    if isinstance(metadata, dict):
        for key in ("src_path", "source_path", "filename"):
            value = metadata.get(key)
            if isinstance(value, str) and os.path.isfile(value):
                return value
    return None


def rewrite_debug_line(artifact, metadata=None, options=None):
    """``npubin``-stage post-processor. Returns ``artifact`` unchanged unless
    ``LLVM_EXTRACT_DI_LOCAL_VARIABLES`` is set. Accepts the stage's ``bytes`` (or
    a path) and returns the rewritten ``bytes`` (or the same path, patched in
    place). Never raises: failures are logged and the artifact is returned as-is.
    """
    if not _enabled():
        return artifact

    src = _resolve_source(metadata)
    try:
        if isinstance(artifact, (bytes, bytearray)):
            new_blob, result = rewrite_debug_line_blob(bytes(artifact), src)
            if result.changed:
                log.info("debug-line: demoted %d stop(s) %s -> %s", result.demoted, result.before, result.after)
            else:
                log.debug("debug-line: no change (%s)", result.reason)
            return new_blob
        if isinstance(artifact, str):
            result = rewrite_debug_line_file(artifact, src_path=src)
            if result.changed:
                log.info("debug-line: demoted %d stop(s) in %s", result.demoted, artifact)
            return artifact
        log.warning("debug-line: unexpected npubin artifact type %r; skipping", type(artifact))
        return artifact
    except Exception:  # never break the build over debug-info cleanup
        log.exception("debug-line: rewrite skipped due to error")
        return artifact


# ── CLI ───────────────────────────────────────────────────────────────────────


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite a .npubin's .debug_line for clean msdebug stepping.")
    parser.add_argument("binary")
    parser.add_argument("--src", help="kernel .py (loop detection + user-file id)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out", help="write patched copy here")
    group.add_argument("--in-place", action="store_true", help="patch in place")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    with open(args.binary, "rb") as handle:
        blob = handle.read()

    if args.dry_run:
        _, result = rewrite_debug_line_blob(blob, args.src)
        log.info("demote %d (dup:%d foreign:%d line0:%d over_src_len:%d special_skipped:%d)", result.demoted,
                 result.counts.get("dup", 0), result.counts.get("foreign", 0), result.counts.get("line0", 0),
                 result.counts.get("over", 0), result.counts.get("special_skip", 0))
        log.info("before: %s", result.before)
        log.info("after:  %s", result.after)
        log.info("(dry run — no write; reason=%s)", result.reason)
        return 0

    out = None if args.in_place else (args.out or args.binary + ".patched.npubin")
    result = rewrite_debug_line_file(args.binary, src_path=args.src, out_path=out)
    if result.changed:
        log.info("demoted %d stop(s): %s -> %s", result.demoted, result.before, result.after)
        log.info("written: %s", out or args.binary)
        if out:
            log.info("install: cp %s %s", out, args.binary)
        return 0
    log.info("no change (%s)", result.reason)
    return 1


if __name__ == "__main__":
    sys.exit(_main())
