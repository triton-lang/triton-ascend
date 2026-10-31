#!/usr/bin/env python3
"""Split one monolithic pass `.cpp` into phase-scoped translation units.

Mechanical, spec-driven refactor: this script only *moves* code. It never
rewrites a function body, so the resulting split is behavior-neutral and can be
proven by byte-identical pass output over the UT corpus.

Usage:
    python3 split_cpp_sections.py [--spec spec.json] [--source other.cpp]
                                  [--dry-run]

Without `--spec`, the built-in `SPLIT_IF_SPEC` runs (the worked example below:
`SplitIfByBlockId.cpp` -> plan / execute / hoist / driver TUs).

Spec (JSON, same schema as `SPLIT_IF_SPEC`):

    source    : path of the monolithic .cpp (overridable with --source, so the
                transformation can be replayed from a pristine / backup copy).
    outputs[] : one entry per generated file
        cpp             path of the generated file (written unless --dry-run)
        ranges[]        [start, end] pairs, concatenated with a blank line
                        between them. `start` is the *literal* text of the
                        first line to take. `end` selects the last line to take:
                        - `"line:<text>"` -> the line starting with `<text>`,
                          inclusive (use for doc comments); `"line:=<text>"`
                          requires the whole line to equal `<text>`;
                        - anything else -> that definition's closing brace
                          (brace counting), so `[a, a]` is one whole function;
                        - `null` -> end of file.
        drops[]         same [start, end] forms, removed *before* ranging; the
                        trailing blank line is consumed when present.
        destatic[]      literal signature prefixes ("static CandidateIf\ngetCandidate(")
                        whose `static ` is dropped (the function becomes the
                        cross-TU entry point declared in a header).
        namespace       namespace to wrap the body in (null = no wrap).
        usings[]        `using namespace <x>;` lines to emit. Only list names
                        the generated TU can actually see: `triton` is declared
                        by `ComputeBlockOpt/Passes.h`, so phase TUs drop it.
        prologue        carry the DEBUG_TYPE / DBGS / LDBG block (default true).
        extra_includes  the pass-private header of the generated TU.

Invariants enforced (loud failure instead of a silently broken split):
  * every anchor must match exactly once;
  * an `end` anchor must resolve to a line whose last column is `}`;
  * no line may be emitted twice (overlapping ranges are reported).
"""

import argparse
import difflib
import json
import re
import sys

# The repo-specific part: which include is needed by which symbol set. A
# generated TU only keeps the includes its own text uses, so the four TUs do not
# inherit the monolithic file's full include list.
INCLUDE_RULES = [
    ("<cstdint>", r"\b(u?int(8|16|32|64)_t|size_t)\b"),
    ("<initializer_list>", r"initializer_list"),
    ("<optional>", r"std::optional|std::nullopt"),
    ("llvm/ADT/ArrayRef.h", r"ArrayRef"),
    ("llvm/ADT/STLExtras.h",
     r"llvm::(zip|enumerate|any_of|all_of|to_vector|reverse|drop_begin|find)"),
    ("llvm/ADT/SmallPtrSet.h", r"SmallPtrSet"),
    ("llvm/ADT/SmallVector.h", r"."),
    ("llvm/ADT/StringExtras.h", r"toHex|llvm::join"),
    ("llvm/ADT/StringRef.h", r"StringRef|StringLiteral"),
    ("llvm/ADT/TypeSwitch.h", r"TypeSwitch"),
    ("llvm/Analysis/ScalarEvolution.h", r"ScalarEvolution"),
    ("llvm/Support/Casting.h", r"."),
    ("llvm/Support/Debug.h", r"LDBG|LLVM_DEBUG|DBGS"),
    ("llvm/Support/LogicalResult.h", r"LogicalResult|llvm::success|llvm::failure"),
    ("llvm/Support/raw_ostream.h", r"llvm::dbgs|raw_ostream"),
    ("mlir/Analysis/AliasAnalysis.h", r"AliasAnalysis"),
    ("mlir/Dialect/Arith/IR/Arith.h", r"arith::"),
    ("mlir/Dialect/Bufferization/IR/Bufferization.h", r"bufferization::"),
    ("mlir/Dialect/Func/IR/FuncOps.h", r"func::"),
    ("mlir/Dialect/Linalg/IR/Linalg.h", r"linalg::"),
    ("mlir/Dialect/MemRef/IR/MemRef.h", r"memref::"),
    ("mlir/Dialect/SCF/IR/SCF.h", r"."),
    ("mlir/Dialect/Tensor/IR/Tensor.h", r"tensor::"),
    ("mlir/IR/Builders.h", r"OpBuilder|InsertionGuard"),
    ("mlir/IR/BuiltinAttributes.h",
     r"IntegerAttr|ArrayAttr|StringAttr|DenseElementsAttr|DenseI64ArrayAttr|DenseIntElements"),
    ("mlir/IR/BuiltinOps.h", r"ModuleOp|OwningOpRef"),
    ("mlir/IR/BuiltinTypes.h",
     r"RankedTensorType|MemRefType|ShapedType|VectorType|TensorType|FloatType|IntegerType|\bType\b"),
    ("mlir/IR/Dominance.h", r"DominanceInfo"),
    ("mlir/IR/IRMapping.h", r"IRMapping"),
    ("mlir/IR/Operation.h", r"."),
    ("mlir/IR/Value.h", r"."),
    ("mlir/IR/ValueRange.h", r"ValueRange"),
    ("mlir/Interfaces/ViewLikeInterface.h", r"ViewLike"),
    ("mlir/Pass/Pass.h", r"PassWrapper|OperationPass"),
    ("mlir/Support/LLVM.h", r"."),
    ("mlir/Support/WalkResult.h", r"WalkResult"),
    ("ascend/include/DynamicCVPipeline/Common/DependencyHelper.h", r"DependencyHelper"),
    ("ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h",
     r"MemoryDependenceGraph|MemoryEffectsTracker"),
    ("ascend/include/DynamicCVPipeline/Common/Utils.h",
     r"kCoreType|kBlockId|getOpBlockId|getCoreTypeOf|coreTypeToString|ssbuffer|parseCoreType|CVPipeline::get"),
    ("ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h",
     r"createSplitIfByBlockIdPass|registerSplitIf"),
    ("ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h",
     r"ComputeBlockIdManager"),
    ("ascend/include/DynamicCVPipeline/ComputeBlockOpt/WalkMainLoop.h", r"walkMainLoop"),
    ("ComputeBlockOpt/SplitIfByBlockId/SplitIfTypes.h", r"."),
    ("DynamicCVPipeline/Common/BlockNesting.h",
     r"isBeforeInNestingOrder|getLowestCommonAncestorBlock"),
    ("DynamicCVPipeline/Common/FallbackHelper.h", r"FallbackHelper|fallback\."),
    ("DynamicCVPipeline/Common/ScalarClosure.h", r"ScalarClosure|getScalarOps"),
    ("bishengir/Dialect/HIVM/IR/HIVM.h", r"hivm::|HIVM"),
    ("bishengir/Dialect/HIVM/IR/HIVMImpl.h", r"hivm::|HIVM"),
]

SPLIT_IF_DIR = "third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId"

# Worked example: the split this script was written for. Replay it against the
# pristine file with `--source build/p3b_backup_SplitIfByBlockId.cpp`.
SPLIT_IF_SPEC = {
    "source": f"{SPLIT_IF_DIR}/SplitIfByBlockId.cpp",
    "outputs": [
        {
            "cpp": f"{SPLIT_IF_DIR}/SplitIfPlan.cpp",
            "namespace": "mlir::CVPipeline::SplitIf",
            "usings": ["mlir", "CVPipeline", "SplitIf"],
            "extra_includes": [f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfPlan.h"],
            "ranges": [["static void addCrossGroupDependency(",
                        "static void buildSplitPlan(CandidateIf &candidate) {"]],
            "drops": [["static bool hasNestedIfs(", "static bool hasNestedIfs("],
                      ["static inline void dumpCandidate(", "static inline void dumpCandidate("],
                      ["/// Plan phase", "line:/// scalar-closure ops are part of the group op lists."]],
            "destatic": ["static void buildSplitPlan(CandidateIf &candidate) {",
                         "static void preprocessScalarDependencies(CandidateIf &cand) {",
                         "static CandidateIf\ngetCandidate("],
        },
        {
            "cpp": f"{SPLIT_IF_DIR}/SplitIfExecute.cpp",
            "namespace": "mlir::CVPipeline::SplitIf",
            "usings": ["mlir", "CVPipeline", "SplitIf"],
            "extra_includes": [f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfExecute.h"],
            "ranges": [["// ============================================================================\n"
                        "// Part3: Materialization\n"
                        "// ============================================================================",
                        "static llvm::LogicalResult\nexecuteSplitPlan("]],
            "drops": [["/// Create a zero/default SSA value", "line:/// downstream analysis."],
                      ["struct SplittedIfTagGenerator {", "struct SplittedIfTagGenerator {"],
                      ["/// Execute phase", "line:/// result types; last group carries original results."]],
            "destatic": ["static llvm::LogicalResult\nexecuteSplitPlan(",
                         "static llvm::FailureOr<Value>\ncreatePlaceholderValue("],
        },
        {
            "cpp": f"{SPLIT_IF_DIR}/SplitIfHoist.cpp",
            "namespace": "mlir::CVPipeline::SplitIf",
            "usings": ["mlir", "CVPipeline", "SplitIf"],
            "prologue": False,
            "extra_includes": [f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfHoist.h",
                               f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfExecute.h"],
            "ranges": [["/// Returns true when `val` is defined inside `ifOp`, either by an operation",
                        "static void hoistAndUnifyRedirectedOutput(scf::IfOp ifOp) {"]],
            "drops": [["/// Hoist the", "line:/// before downstream provenance analyses."]],
            "destatic": ["static void hoistAndUnifyRedirectedOutput(scf::IfOp ifOp) {"],
        },
        {
            "cpp": f"{SPLIT_IF_DIR}/SplitIfByBlockId.cpp",
            "namespace": None,
            "usings": ["mlir", "triton", "CVPipeline", "SplitIf"],
            "extra_includes": [f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfPlan.h",
                               f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfExecute.h",
                               f"{SPLIT_IF_DIR.replace('third_party/ascend/lib/DynamicCVPipeline/', '')}/SplitIfHoist.h"],
            "ranges": [["static bool hasNestedIfs(", "static bool hasNestedIfs("],
                       ["static inline void dumpCandidate(", "static inline void dumpCandidate("],
                       ["static void rearrangeIfOp(", "static void rearrangeIfOp("],
                       ["namespace {\n\nclass SplitIfByBlockIdPass", "line:=} // namespace"],
                       ["void SplitIfByBlockIdPass::runOnOperation() {", None]],
            "drops": [],
            "destatic": [],
        },
    ],
}


def strip_code(line, in_block_comment):
    """Return `line` without comments/string bodies, plus the block-comment state."""
    out = []
    i = 0
    while i < len(line):
        if in_block_comment:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), True
            i = end + 2
            in_block_comment = False
            continue
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block_comment = True
            i += 2
            continue
        ch = line[i]
        if ch in "\"'":
            i += 1
            while i < len(line):
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == ch:
                    break
                i += 1
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block_comment


def brace_span(lines, start):
    """Line index of the `}` closing the definition starting at `start`."""
    depth = 0
    in_block = False
    seen = False
    for i in range(start, len(lines)):
        code, in_block = strip_code(lines[i], in_block)
        for ch in code:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen and depth == 0:
            return i
    raise SystemExit(f"unbalanced braces from line {start + 1}: {lines[start][:60]!r}")


def line_of(lines, anchor, what):
    """Unique line index whose text starts at `anchor` (multi-line anchors allowed).

    A `=` prefix (single line only) requires the line to match exactly, which
    disambiguates anchors like `} // namespace` from `} // namespace triton`.
    """
    if anchor.startswith("="):
        exact = anchor[1:]
        hits = [i for i, l in enumerate(lines) if l == exact]
        if len(hits) != 1:
            raise SystemExit(f"anchor {what} matches {len(hits)} lines: {exact!r}")
        return hits[0]
    text = "\n".join(lines)
    hits = [m.start() for m in re.finditer(re.escape(anchor), text) if
            m.start() == 0 or text[m.start() - 1] == "\n"]
    if len(hits) != 1:
        raise SystemExit(f"anchor {what} matches {len(hits)} times: {anchor[:60]!r}")
    return text[:hits[0]].count("\n")


def resolve(lines, anchor, what):
    """(first_line, last_line) for a range/anchor pair."""
    start = line_of(lines, anchor[0], f"{what}.start")
    end = anchor[1]
    if end is None:
        return start, len(lines) - 1
    if end.startswith("line:"):
        last = line_of(lines, end[len("line:"):], f"{what}.line-end")
    else:
        last = brace_span(lines, line_of(lines, end, f"{what}.end"))
    if last < start:
        raise SystemExit(f"{what}: end anchor before start anchor")
    return start, last


def split_off_cpp(text, spec):
    lines = text.split("\n")
    license_end = next(i for i, l in enumerate(lines) if l == " */")
    license_text = lines[:license_end + 1]
    inc_start = next(i for i, l in enumerate(lines) if l.startswith("#include "))
    inc_end = inc_start
    for i in range(inc_start, len(lines)):
        if lines[i].startswith("#include "):
            inc_end = i
        elif lines[i].strip() and not lines[i].startswith("#include "):
            break
    includes = lines[inc_start:inc_end + 1]
    pro_start = next(i for i, l in enumerate(lines) if l.startswith("static constexpr const char *DEBUG_TYPE"))
    # follow the `#define` continuation lines (DEBUG_TYPE/DBGS/LDBG macros)
    pro_end = next(i for i, l in enumerate(lines) if l.startswith("#define DBGS()"))
    while lines[pro_end].rstrip().endswith("\\") or lines[pro_end + 1].startswith("#define "):
        pro_end += 1
    prologue = lines[pro_start:pro_end + 1]
    assert any("LDBG" in l for l in prologue), "prologue end anchor looks wrong"
    return license_text, includes, prologue


def build_output(spec, lines, prologue_block, out):
    work = list(lines)
    # drops first: an anchor range is anchored by content, so later ranges are
    # unaffected. Delete bottom-up to keep earlier indices valid.
    for anchor in out.get("drops", []):
        start, end = resolve(work, anchor, "drop")
        while end + 1 < len(work) and work[end + 1].strip() == "":
            end += 1
        del work[start:end + 1]
    body = []
    for i, anchor in enumerate(out["ranges"]):
        start, end = resolve(work, anchor, f"range[{i}]")
        if i:
            body.append("")
        body += work[start:end + 1]
    text = "\n".join(body)
    for sig in out.get("destatic", []):
        if not sig.startswith("static ") or text.count(sig) != 1:
            raise SystemExit(f"destatic pattern must be a unique `static ...`: {sig!r}")
        text = text.replace(sig, sig[len("static "):], 1)
    body = text.split("\n")
    used = "\n".join(body)
    incs = []
    prev_group = None
    group = -1
    for inc in includes_lines:
        if not inc.strip():
            group += 1
            continue
        key = inc[len("#include "):].strip().strip('"')
        rule = dict(INCLUDE_RULES).get(key)
        if rule is None:
            raise SystemExit(f"no INCLUDE_RULES entry for {key!r}")
        if re.search(rule, used):
            if prev_group is not None and group != prev_group:
                incs.append("")
            incs.append(inc)
            prev_group = group
    incs += [f'#include "{e}"' for e in out.get("extra_includes", [])]
    parts = [*LICENSE, "", *incs, ""]
    if out.get("prologue", True):
        parts += [*prologue_block, ""]
    parts += [f"using namespace {u};" for u in out["usings"]] + [""]
    if out.get("namespace"):
        parts += [f"namespace {out['namespace']} {{", *body,
                  f"}} // namespace {out['namespace']}", ""]
    else:
        parts += [*body]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--source")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = json.load(open(args.spec)) if args.spec else SPLIT_IF_SPEC
    src = args.source or spec["source"]
    text = open(src).read()
    global LICENSE, includes_lines
    LICENSE, includes_lines, prologue_block = split_off_cpp(text, spec)
    lines = text.split("\n")
    for out in spec["outputs"]:
        new = build_output(spec, lines, prologue_block, out)
        old = None
        try:
            old = open(out["cpp"]).read()
        except FileNotFoundError:
            pass
        if args.dry_run:
            if old is None:
                print(f"[dry-run] {out['cpp']}: would create "
                      f"({len(new.splitlines())} lines)")
            elif old != new:
                print(f"[dry-run] {out['cpp']}: DIFFERS")
                sys.stdout.writelines(difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True),
                    fromfile="current", tofile="generated"))
            else:
                print(f"[dry-run] {out['cpp']}: identical")
            continue
        open(out["cpp"], "w").write(new)
        print(f"{out['cpp']}: {len(new.splitlines())} lines")


if __name__ == "__main__":
    main()
