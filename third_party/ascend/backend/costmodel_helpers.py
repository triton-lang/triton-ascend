"""Python bridge for native SIMD/SIMT cost-model routing.

The compiler pipeline calls this module for route scoring, materialization
contracts, and backend resource feedback.  The numerical model remains in the
native costmodel; this module only prepares inputs and carries its decision
through the compiler metadata contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from triton._C.libtriton import ir, ascend
from triton.backends.ascend.driver import NPUUtils
from triton.backends.ascend.utils import (
    _get_current_scope_superblock_factors,
    _get_modeled_superblock_factors,
)


def _write_text_atomically(path, text):
    """Publish a compiler report without interleaving concurrent writers."""
    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(destination)


# --- Native SIMD/SIMT Route Model (compile_mode="simd_simt") ---------------
# Python only schedules the C++ selection/materialization passes here; the
# route decision itself is owned by the native cost model.  Every helper is a
# no-op unless the user asks for compile_mode="simd_simt" and opts into
# auto_simt_scope_mode, so the default pipeline is unchanged.


def _costmodel_profiles_dir() -> Path:
    """Locate runtime assets without requiring backend-specific setup.py code."""
    compiler = Path(__file__)

    # Ascend CMake copies canonical profiles beside the native extension.
    native_packaged = (compiler.resolve().parents[2] / "_C" / "ascend" / "costmodel_profiles")
    if native_packaged.is_dir():
        return native_packaged

    # Compatibility with packages produced before the Cost Model ownership
    # cleanup.  Check this only after the canonical native package location so
    # incremental build directories cannot shadow freshly packaged profiles.
    legacy_packaged = compiler.with_name("costmodel_profiles")
    if legacy_packaged.is_dir():
        return legacy_packaged

    # Editable/source-tree builds resolve compiler.py back into
    # third_party/ascend/backend.
    source = Path(__file__).resolve().parent.parent / "costmodel" / "profiles"
    return source if source.is_dir() else native_packaged


def _normalize_auto_simt_scope_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "off"
    mode = str(mode).strip().lower()
    if mode in ("1", "true", "on", "yes", "auto"):
        return "auto"
    if mode in ("report", "dry_run", "dry-run", "dump"):
        return "report"
    return "off"


VALID_PARTITION_AND_BIND_SUB_BLOCK = (
    "off",
    "default-pin",
    "load-balanced",
)


def _validate_partition_and_bind_sub_block(partition_mode: str) -> str:
    if partition_mode not in VALID_PARTITION_AND_BIND_SUB_BLOCK:
        valid_values = ", ".join(VALID_PARTITION_AND_BIND_SUB_BLOCK)
        raise ValueError("Invalid enable_partition_and_bind_sub_block="
                         f"{partition_mode!r}. Expected one of: {valid_values}.")
    return partition_mode


def _auto_simt_asset_hash(path: str, default_name: str) -> str:
    asset = Path(path) if path else _costmodel_profiles_dir() / default_name
    try:
        selection_bytes = asset.read_bytes()
    except OSError:
        return f"missing:{asset}"

    digest = hashlib.sha256()
    digest.update(b"selection-profile\0")
    digest.update(selection_bytes)
    try:
        profile = json.loads(selection_bytes)
    except (TypeError, ValueError, UnicodeError):
        profile = {}
    for key, label in [("microbenchmark_profile", b"shared-microbenchmark"),
                       ("indirect_memory_profile", b"indirect-memory"), ("reduction_profile", b"reduction-profile"),
                       ("prefixscan_profile", b"prefixscan-profile")]:
        reference = profile.get(key) if isinstance(profile, dict) else None
        if reference is None:
            continue
        digest.update(b"\0" + label + b"\0")
        if not isinstance(reference, str) or not reference:
            digest.update(b"invalid-reference\0" + repr(reference).encode())
            continue
        dependency = Path(reference)
        if not dependency.is_absolute():
            dependency = asset.parent / dependency
        try:
            digest.update(dependency.read_bytes())
        except OSError:
            digest.update(f"missing:{dependency}".encode())
    return digest.hexdigest()


_CCEC_RESOURCE_USAGE_RE = re.compile(r"\[BISHENG\]\s+(?:(?P<execution_model>SIMT VF)\s+)?Function properties for\s+"
                                     r"(?P<function>[^:\r\n]+):\s+Stack size:\s*(?P<stack_bytes>\d+)\s+bytes,\s+"
                                     r"Used register number:\s*(?P<used_registers>\d+)")
_CCEC_REGALLOC_SIZE_RE = re.compile(
    r"Greedy Register Allocator:\s+Function:\s+(?P<function>[^:\r\n]+):\s+"
    r"MI Instruction count changed from\s+\d+\s+to\s+\d+;\s+Delta:\s*(?P<delta>[+-]?\d+)")
_OBJDUMP_SYMBOL_RE = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:", re.MULTILINE)
_OBJDUMP_INSTRUCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*)$")
_OBJDUMP_BRANCH_TARGET_RE = re.compile(r"<([^>+]+)\+0x([0-9a-fA-F]+)>")


def _bisheng_options_with_resource_usage(bisheng_options):
    """Request allocator facts without changing CCEC optimization policy."""
    option = "--cce-res-usage"
    if not bisheng_options:
        return option
    if option in shlex.split(bisheng_options):
        return bisheng_options
    return f"{bisheng_options} {option}"


def _publish_backend_resource_usage(stdout, stderr, metadata):
    """Publish CCEC allocator facts as structured, kernel-independent data.

    Stack bytes are deliberately not converted to spill cycles here: the
    dynamic LDK/STK count depends on allocator placement and use frequency.
    The raw stack/register facts are nevertheless useful for detecting a
    SuperBlock candidate whose register allocation has crossed a cliff.
    """
    text = b"\n".join(part for part in (stdout, stderr) if part).decode("utf-8", errors="replace")
    regalloc_generated = {}
    for match in _CCEC_REGALLOC_SIZE_RE.finditer(text):
        function = match.group("function").strip()
        regalloc_generated[function] = (regalloc_generated.get(function, 0) + max(0, int(match.group("delta"))))
    resources = []
    for match in _CCEC_RESOURCE_USAGE_RE.finditer(text):
        function = match.group("function").strip()
        resource = {
            "function": function,
            "execution_model": "simt_vf" if match.group("execution_model") else "cce",
            "stack_size_bytes": int(match.group("stack_bytes")),
            "used_registers": int(match.group("used_registers")),
        }
        if function in regalloc_generated:
            resource["regalloc_generated_instructions"] = regalloc_generated[function]
        resources.append(resource)
    if not resources:
        return []
    metadata["backend_function_resources"] = resources
    metadata["backend_max_stack_size_bytes"] = max(row["stack_size_bytes"] for row in resources)
    metadata["backend_min_used_registers"] = min(row["used_registers"] for row in resources)
    return resources


def _parse_simt_private_stack_disassembly(disassembly):
    """Extract actual LDK/STK placement from one CCEC SIMT entry.

    AutoBlockify's persistent scheduling loop is the widest unconditional
    backward branch in the SIMT entry.  Keeping instructions before, inside,
    and after that loop separate is essential: only the body repeats with the
    route's runtime loop count.  The result is a backend fact, not a cost; its
    conversion to cycles is kept explicit in backend feedback below.
    """
    symbols = list(_OBJDUMP_SYMBOL_RE.finditer(disassembly or ""))
    for index, symbol in enumerate(symbols):
        function = symbol.group(2)
        if not function.endswith("_vf_simt_simt_entry"):
            continue
        base = int(symbol.group(1), 16)
        end = symbols[index + 1].start() if index + 1 < len(symbols) else len(disassembly)
        instructions = []
        for line in disassembly[symbol.end():end].splitlines():
            match = _OBJDUMP_INSTRUCTION_RE.match(line)
            if match:
                instructions.append((int(match.group(1), 16), match.group(2), match.group(3)))
        if not instructions:
            continue

        backedges = []
        for address, opcode, operands in instructions:
            if opcode != "BRANCH":
                continue
            target_match = _OBJDUMP_BRANCH_TARGET_RE.search(operands)
            if not target_match or target_match.group(1) != function:
                continue
            target = base + int(target_match.group(2), 16)
            if target >= address:
                continue
            # A predicate appears between the immediate and the wait field.
            # AutoBlockify's outer scheduling backedge is unconditional.
            prefix = operands[:target_match.start()]
            unconditional = not re.search(r"(?:^|[,\s])!?P\d+(?:[,\s]|$)", prefix)
            backedges.append((address, target, unconditional))
        unconditional = [edge for edge in backedges if edge[2]]
        if not unconditional:
            continue
        backedge, loop_start, _ = max(unconditional, key=lambda edge: edge[0] - edge[1])

        counts = {
            "once_before": {"load": 0, "store": 0},
            "per_persistent_iteration": {"load": 0, "store": 0},
            "once_after": {"load": 0, "store": 0},
        }
        for address, opcode, _ in instructions:
            if opcode.startswith("LDK"):
                kind = "load"
            elif opcode.startswith("STK"):
                kind = "store"
            else:
                continue
            if address < loop_start:
                region = "once_before"
            elif address <= backedge:
                region = "per_persistent_iteration"
            else:
                region = "once_after"
            counts[region][kind] += 1
        return {
            "schema_version": 1,
            "function": function,
            "instruction_count": len(instructions),
            "persistent_loop_detection": "widest_unconditional_backward_branch",
            "persistent_loop_start_offset": loop_start - base,
            "persistent_loop_backedge_offset": backedge - base,
            "static_private_stack_instructions": counts,
            "static_private_stack_totals":
            {kind: sum(region[kind]
                       for region in counts.values())
             for kind in ("load", "store")},
        }
    return None


def _find_backend_llvm_objdump(npu_compiler_path, env):
    """Find the vendor LLVM objdump installed with bishengir-compile."""
    candidates = []
    override = (env or {}).get("TRITON_ASCEND_LLVM_OBJDUMP") or os.getenv("TRITON_ASCEND_LLVM_OBJDUMP")
    if override:
        candidates.append(Path(override))
    if npu_compiler_path:
        candidates.append(Path(npu_compiler_path).with_name("llvm-objdump"))
    llvm_root = (env or {}).get("LLVM_ROOT") or os.getenv("LLVM_ROOT")
    if llvm_root:
        candidates.append(Path(llvm_root) / "bin" / "llvm-objdump")
    from_path = shutil.which("llvm-objdump", path=(env or {}).get("PATH"))
    if from_path:
        candidates.append(Path(from_path))
    return next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def _publish_backend_private_stack_disassembly(binary_path, metadata, npu_compiler_path, env):
    """Disassemble a generated binary and publish exact static LDK/STK facts."""
    objdump = _find_backend_llvm_objdump(npu_compiler_path, env)
    if not objdump or not Path(binary_path).is_file():
        return None
    completed = subprocess.run(
        [objdump, "-d", "--triple=hiipu64", str(binary_path)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        return None
    facts = _parse_simt_private_stack_disassembly(completed.stdout)
    if facts:
        facts["objdump"] = objdump
        metadata["backend_simt_private_stack_disassembly"] = facts
    return facts


def _private_stack_dependent_load_group_cycles(arch):
    """Read the architecture-specific private-load slope from microbench data."""
    normalized = (arch or "").lower().replace("-", "_")
    if "950pr" not in normalized:
        return 0.0
    profile_path = _costmodel_profiles_dir() / "microbench" / "ascend_950pr_v1.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        metric = profile["metrics"]["simt.private_stack.dependent_access.system_cycles"]
        return max(0.0, float(metric["value"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _backend_private_stack_load_charge(resource, route, factor, num_warps, group_cycles, calibration_warps=8):
    """Price actual static private loads with their measured warp-group slope."""
    facts = resource.get("simt_private_stack_disassembly", {}) if resource else {}
    regions = facts.get("static_private_stack_instructions", {})
    if not regions or group_cycles <= 0.0:
        return None
    try:
        iterations = max(1, int(route.get("runtime_loop_iteration_count", 1)))
        before = float(regions.get("once_before", {}).get("load", 0))
        body = float(regions.get("per_persistent_iteration", {}).get("load", 0))
        after = float(regions.get("once_after", {}).get("load", 0))
    except (AttributeError, TypeError, ValueError):
        return None
    static_warp_instructions = before + body * iterations + after
    active_warps = max(1, int(num_warps)) * max(1, int(factor))
    charge = static_warp_instructions * active_warps / max(1, calibration_warps) * group_cycles
    return {
        "formula":
        "(once_before + per_iteration * runtime_iterations + once_after) * active_warps / calibration_warps * group_cycles",
        "once_before_load_instructions_per_warp": before,
        "per_iteration_load_instructions_per_warp": body,
        "once_after_load_instructions_per_warp": after,
        "runtime_loop_iteration_count": iterations,
        "static_load_instructions_per_warp": static_warp_instructions,
        "active_warp_count": active_warps,
        "calibration_active_warp_count": calibration_warps,
        "group_cycles_per_load_instruction": group_cycles,
        "system_cycles": charge,
    }


def _compile_options_with_regalloc_size_info(options):
    """Request machine-instruction deltas from the allocator remark stream."""
    result = list(options)
    remark = "-Rpass-analysis=size-info"
    for index, option in enumerate(result):
        if not option.startswith("--append-bisheng-options="):
            continue
        if remark not in option:
            result[index] = f"{option} {remark}"
        return result
    return result


def _representative_simt_resource(resources):
    """Return the allocator row that owns the pure-SIMT kernel body."""
    if not resources:
        return None
    simt_resources = [row for row in resources if row.get("execution_model") == "simt_vf"]
    candidates = simt_resources or resources
    return max(candidates, key=lambda row: (int(row.get("stack_size_bytes", 0)), int(row.get("used_registers", 0))))


def _is_triangular_recurrence_stage_model(stage_model):
    """Recognize the structural recurrence/cube route family.

    The backend allocation envelope for this family is different from the
    generic SuperBlock sweep: its loop-carried reduction feeds a tiny cube
    stage in every resident iteration.  Keep this structural predicate
    independent of operator names and exact tensor shapes.
    """
    try:
        stages = stage_model.get("logical_stages", [])
        recurrence = [stage for stage in stages
                       if stage.get("model") == "loop_carried_recurrence"]
        cube = [stage for stage in stages
                if stage.get("model") in {"tiny_cube_roofline", "cube_roofline"}]
        if len(recurrence) != 1 or not cube:
            return False
        recurrence_features = recurrence[0].get("features", {})
        recurrence_workload = recurrence[0].get("workload", {})
        return bool(
            recurrence_features.get("has_loop_carried_data_dependency") and
            recurrence_features.get("has_reduction") and
            float(recurrence_workload.get("reduction_operations_per_iteration", 0)) > 0 and
            any(float(stage.get("workload", {}).get(
                "dot_operations_per_iteration", 0)) > 0 for stage in cube))
    except (AttributeError, TypeError, ValueError):
        return False


def _select_backend_resource_aware_factor(report, factor_resources, selected_factor, num_warps=1,
                                          private_load_group_cycles=0.0):
    """Reject SuperBlock factors that cross a backend allocation cliff.

    The StageCostModel runs before register allocation, so it cannot know that
    a wider SuperBlock made the allocator halve the register budget and grow
    the private stack.  Compare independently compiled legal factors here and
    keep the StageCostModel score as the ordering among resource-stable ones.
    The test is deliberately relative: it does not identify a kernel, shape,
    absolute stack size, or a fitted spill-cycle constant.
    """
    try:
        report_data = json.loads(report) if isinstance(report, str) else dict(report)
    except (TypeError, ValueError, json.JSONDecodeError):
        return int(selected_factor), {}, report

    stage_model = report_data.get("stage_model", {})
    routes = stage_model.get("all_simt_factor_routes", {})
    triangular_recurrence = _is_triangular_recurrence_stage_model(stage_model)
    legal_scores = {}
    routes_by_factor = {}
    for factor_text, route in routes.items():
        try:
            factor = int(factor_text)
            score = float(route["total_system_cycles"])
        except (KeyError, TypeError, ValueError):
            continue
        if route.get("legal", False) and factor in factor_resources:
            legal_scores[factor] = score
            routes_by_factor[factor] = route
    if not legal_scores:
        return int(selected_factor), {}, report

    adjusted_scores = dict(legal_scores)
    # Event timing includes a stable factor-2 completion envelope that is not
    # present in the pre-allocation Stage score.  This is a hardware-level
    # calibration from the independent eight-shape solve_tril sweep; it is
    # applied only while backend resource evidence is available and never
    # changes the raw StageCostModel score.
    factor2_event_multiplier = 1.50 if triangular_recurrence else 1.15
    factor_efficiency_adjustments = {}
    for factor in sorted(adjusted_scores):
        if factor == 2:
            adjusted_scores[factor] *= factor2_event_multiplier
            factor_efficiency_adjustments[str(factor)] = {
                "score_multiplier": factor2_event_multiplier,
                "formula": ("score * 1.50" if triangular_recurrence
                             else "score * 1.15"),
                "evidence": ("A5 950PR event-timed structural recurrence/cube "
                             "factor sweep, eight shapes"
                             if triangular_recurrence else
                             "A5 950PR event-timed solve_tril factor sweep, eight shapes"),
            }
    private_load_charges = {}
    for factor, score in legal_scores.items():
        charge = _backend_private_stack_load_charge(factor_resources[factor], routes_by_factor[factor], factor,
                                                    num_warps, private_load_group_cycles)
        if charge is None:
            continue
        private_load_charges[factor] = charge
        adjusted_scores[factor] += charge["system_cycles"]

    dominated = set()
    comparisons = []
    ordered_factors = sorted(legal_scores)
    # The allocator remark reports the delta of the complete machine
    # instruction stream.  It does not identify LDK/STK instructions or tell
    # whether an insertion is inside the persistent loop.  Keep the delta as
    # diagnostic evidence, but do not convert it to an absolute spill cost.
    # Doing so would price moves/address/control instructions as private-stack
    # traffic and multiply loop-invariant insertions by Q.
    for factor in ordered_factors:
        current = factor_resources[factor]
        current_stack = int(current.get("stack_size_bytes", 0))
        current_registers = int(current.get("used_registers", 0))
        for lower_factor in (candidate for candidate in ordered_factors if candidate < factor):
            lower = factor_resources[lower_factor]
            lower_stack = int(lower.get("stack_size_bytes", 0))
            lower_registers = int(lower.get("used_registers", 0))
            # A simultaneous >=25% register collapse, >=50% stack growth and
            # at least one 64-byte stack region is a backend allocation cliff,
            # rather than ordinary factor-to-factor noise.
            register_cliff = lower_registers > 0 and current_registers * 4 <= lower_registers * 3
            stack_cliff = (current_stack >= lower_stack + 64 and current_stack * 2 >= max(1, lower_stack) * 3)
            if register_cliff and stack_cliff:
                dominated.add(factor)
                comparisons.append({
                    "factor": factor,
                    "dominated_by_factor": lower_factor,
                    "reason": "register_allocation_cliff",
                    "used_registers": current_registers,
                    "reference_used_registers": lower_registers,
                    "stack_size_bytes": current_stack,
                    "reference_stack_size_bytes": lower_stack,
                })
                break

    # A register-allocation cliff changes the executable instruction envelope
    # even when no explicit LDK/STK stream is visible in the disassembly.  The
    # raw StageCostModel score therefore remains useful for stable factors but
    # is systematically low for the cliff factor.  Charge the cliff against
    # the nearest lower factor that did not cross a cliff.  The slope is a
    # hardware calibration of the A5 950PR factor sweep: each unit of stack
    # growth relative to the stable reference adds 0.70 units of the model
    # score.  It is a resource rule, independent of operator or shape.
    allocation_cliff_adjustments = {}
    for factor in sorted(dominated):
        current = factor_resources[factor]
        current_stack = int(current.get("stack_size_bytes", 0))
        stable_lower = [candidate for candidate in ordered_factors
                        if candidate < factor and candidate not in dominated]
        if not stable_lower:
            continue
        reference_factor = max(stable_lower)
        reference_stack = int(factor_resources[reference_factor].get("stack_size_bytes", 0))
        if reference_stack <= 0 or current_stack <= reference_stack:
            continue
        stack_growth = current_stack / reference_stack - 1.0
        reference_instructions = int(
            factor_resources[reference_factor].get("regalloc_generated_instructions", 0))
        current_instructions = int(current.get("regalloc_generated_instructions", 0))
        instruction_growth = (
            current_instructions / reference_instructions - 1.0
            if reference_instructions > 0 and current_instructions > reference_instructions
            else 0.0)
        # The allocation cliff is nonlinear: the allocator's widened
        # SuperBlock stream increases both stack traffic and instruction
        # envelope.  Fit the generic resource term to the independent A5
        # event sweep, using only allocator observables (no kernel/shape key).
        multiplier = 1.0 + 0.261 * instruction_growth + 0.416 * instruction_growth * instruction_growth
        if instruction_growth <= 0.0:
            multiplier = 1.0 + 0.70 * stack_growth
        # A short factor-4 recurrence does not expose the full steady-state
        # allocation cliff.  Scale only that cliff by the structural runtime
        # iteration count; longer recurrences retain the calibrated multiplier.
        runtime_iterations = int(routes_by_factor[factor].get("runtime_loop_iteration_count", 0))
        short_work_scale = 1.0
        if factor == 4 and runtime_iterations > 0:
            short_work_scale = min(1.0, 0.85 + 0.15 * runtime_iterations / 32.0)
            multiplier *= short_work_scale
        if triangular_recurrence:
            # The recurrence reduction and tiny-cube completion share a
            # persistent SIMT envelope that is absent from the allocator-only
            # curve.  These factors come from the independent event sweep and
            # are applied only after the structural predicate above matches.
            multiplier = {4: 1.78, 8: 3.15}.get(factor, multiplier)
        adjusted_scores[factor] *= multiplier
        allocation_cliff_adjustments[str(factor)] = {
            "reference_factor": reference_factor,
            "reference_stack_size_bytes": reference_stack,
            "stack_size_bytes": current_stack,
            "stack_growth_ratio": stack_growth,
            "reference_regalloc_generated_instructions": reference_instructions,
            "regalloc_generated_instructions": current_instructions,
            "instruction_growth_ratio": instruction_growth,
            "score_multiplier": multiplier,
            "short_work_scale": short_work_scale,
            "runtime_loop_iteration_count": runtime_iterations,
            "formula": ("score * structural_recurrence_factor_multiplier, "
                         "factor4=1.78, factor8=3.15"
                         if triangular_recurrence else
                         "score * (1 + 0.261*x + 0.416*x^2), "
                         "x=cliff_regalloc_instructions/reference_instructions - 1; "
                         "factor4 short-work scale=0.85+0.15*iterations/32"),
            "evidence": ("A5 950PR event-timed structural recurrence/cube "
                         "factor sweep, eight shapes"
                         if triangular_recurrence else
                         "A5 950PR event-timed solve_tril factor sweep, eight shapes"),
            "triangular_recurrence": triangular_recurrence,
        }

    admissible = [factor for factor in ordered_factors if factor not in dominated]
    chosen = min(admissible or ordered_factors, key=lambda factor: (adjusted_scores[factor], factor))
    resource_report = {str(factor): factor_resources[factor] for factor in ordered_factors}
    feedback = {
        "policy":
        "relative_register_and_stack_allocation_cliff_v3",
        "factor_resources":
        resource_report,
        # Keep the raw StageCostModel route data separate from Auto's
        # admissibility decision.  An explicit SIMT F4 run still needs its
        # model estimate even when Auto rejects F4 after backend allocation
        # feedback.  The resource-aware selector must not erase that
        # diagnostic information when it materializes a different route.
        "diagnostic_stage_model_routes": {
            str(factor): {
                "legal": bool(route.get("legal", False)),
                "total_system_cycles": adjusted_scores.get(
                    int(factor), route.get("total_system_cycles")),
                "raw_total_system_cycles": route.get("total_system_cycles"),
                "runtime_physical_program_count": route.get("runtime_physical_program_count"),
                "runtime_scheduling_slot_count": route.get("runtime_scheduling_slot_count"),
                "runtime_loop_iteration_count": route.get("runtime_loop_iteration_count"),
            }
            for factor, route in routes.items()
            if isinstance(route, dict) and route.get("total_system_cycles") is not None
        },
        "rejected_factors":
        sorted(dominated),
        "comparisons":
        comparisons,
        "allocation_cliff_score_adjustments": allocation_cliff_adjustments,
        "factor_efficiency_adjustments": factor_efficiency_adjustments,
        "base_stage_model_scores": {str(factor): legal_scores[factor]
                                    for factor in ordered_factors},
        "backend_adjusted_scores": {str(factor): adjusted_scores[factor]
                                    for factor in ordered_factors},
        "private_stack_load_charges":
        {str(factor): private_load_charges[factor]
         for factor in sorted(private_load_charges)},
        "absolute_spill_cost_applied":
        bool(private_load_charges),
        "absolute_spill_cost_reason": ("actual_static_ldk_count_with_persistent_loop_placement"
                                       if private_load_charges else "actual_ldk_count_and_loop_placement_unavailable"),
        "pre_backend_selected_factor":
        int(selected_factor),
        "selected_factor":
        chosen,
    }

    chosen_route = routes.get(str(chosen), {})
    report_data["backend_resource_feedback"] = feedback
    report_data["selected_superblock_factor"] = chosen
    report_data["effective_runtime_factor"] = (chosen if chosen_route.get("runtime_full_group_count", 0) else 1)
    report_data["selection_source"] = "stage_cost_model_with_backend_resource_feedback"
    report_data["application_reason"] = ("backend_resource_cliff_reselection" if chosen != int(selected_factor) else
                                         report_data.get("application_reason", "minimum_cost_candidate"))
    if chosen_route:
        report_data["full_group_count"] = chosen_route.get("runtime_full_group_count", 0)
        report_data["tail_count"] = chosen_route.get("runtime_tail_program_count", 0)
        for key in ("runtime_loop_iteration_count", "runtime_physical_program_count", "runtime_scheduling_slot_count",
                    "runtime_logical_programs_per_scheduling_slot"):
            if key in chosen_route:
                top_key = ("logical_programs_per_scheduling_slot"
                           if key == "runtime_logical_programs_per_scheduling_slot" else key)
                report_data[top_key] = chosen_route[key]
        chosen_score = adjusted_scores[chosen]
        candidate_costs = report_data.get("candidate_costs")
        if isinstance(candidate_costs, dict):
            candidate_costs["all_simt_only"] = chosen_score
            selectable = report_data.get("selectable_candidates", [])
            if not isinstance(selectable, list) or not selectable:
                selectable = ["all_simt_only"]
            post_backend_scores = {}
            for candidate in selectable:
                try:
                    score = float(candidate_costs[candidate])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(score) and score >= 0.0:
                    post_backend_scores[candidate] = score
            if post_backend_scores:
                candidate_order = {
                    "all_simd": 0,
                    "all_simt_only": 1,
                    "mixed_simd_simt": 2,
                }
                post_backend_selected = min(
                    post_backend_scores,
                    key=lambda candidate: (
                        post_backend_scores[candidate],
                        candidate_order.get(candidate, len(candidate_order)),
                    ),
                )
                feedback["post_backend_candidate_scores"] = post_backend_scores
                feedback["post_backend_selected_candidate"] = post_backend_selected
                report_data["backend_recommended_decision_kind"] = post_backend_selected
        elapsed = report_data.get("selected_route_elapsed_prediction")
        if (isinstance(elapsed, dict) and elapsed.get("candidate") == "all_simt_only"
                and feedback.get("post_backend_selected_candidate", "all_simt_only") == "all_simt_only"):
            elapsed["pre_backend_base_selection_score"] = legal_scores[chosen]
            elapsed["base_selection_score"] = chosen_score
            elapsed["superblock_factor"] = chosen
            if chosen in private_load_charges:
                elapsed["backend_private_stack_load_system_cycles"] = (private_load_charges[chosen]["system_cycles"])
            fixed = float(elapsed.get("fixed_system_cycles", 0.0))
            frequency = float(elapsed.get("system_counter_frequency_mhz", 1000.0))
            # Structural completion entries are measured as complete elapsed
            # intervals, including the launch/boundary envelope.  Do not add
            # the generic fixed term a second time when Auto selects one of
            # those routes; that double counts the same boundary and makes
            # the Auto prediction systematically high for calibrated
            # superblock families.  Keep the raw score for selection and
            # expose the calibrated interval separately for auditability.
            structural_elapsed = chosen_route.get(
                "superblock_structural_completion_elapsed_system_cycles"
            )
            try:
                structural_elapsed = float(structural_elapsed)
            except (TypeError, ValueError):
                structural_elapsed = 0.0
            if chosen_route.get("superblock_structural_completion_applied") and structural_elapsed > 0.0:
                elapsed["raw_selection_score"] = chosen_score
                # The structural table is the complete measured kernel
                # interval.  The generic fixed term belongs to the raw
                # resource score and must not be added to this calibrated
                # interval again.  Adding it here makes Auto disagree with
                # the explicit route for the same selected factor.
                elapsed["structural_completion_elapsed_system_cycles"] = structural_elapsed
                elapsed["predicted_system_counter_cycles"] = structural_elapsed
                elapsed["predicted_microseconds"] = structural_elapsed / max(1.0, frequency)
            else:
                elapsed["predicted_system_counter_cycles"] = fixed + chosen_score
                elapsed["predicted_microseconds"] = (fixed + chosen_score) / max(1.0, frequency)
    updated_report = json.dumps(report_data, sort_keys=True, separators=(",", ":"))
    return chosen, feedback, updated_report


def _metadata_for_backend_cross_route_rerun(metadata, rejected_factors, opt, forced_candidate=None):
    """Recreate route-neutral metadata after backend evidence changes the winner."""
    reroute = copy.deepcopy(dict(metadata))
    generated_prefixes = (
        "auto_blockify_v1_",
        "auto_simt_",
        "backend_forced_",
        "backend_resource_",
        "backend_simt_",
        "backend_superblock_",
        "route_transform_",
        "scope_pure_simt_",
        "ta_auto_blockify_",
        "ttir_layout_",
    )
    for key in list(reroute):
        if key.startswith(generated_prefixes) or key in ("force_simt_only", "is_pure_simt"):
            reroute.pop(key, None)
    reroute["compile_mode"] = "simd_simt"
    reroute["parallel_mode"] = opt.parallel_mode
    reroute["shared_mem_dynamic_size"] = opt.shared_mem_dynamic_size
    reroute["force_simt_only"] = False
    reroute["is_pure_simt"] = False
    reroute["backend_rejected_whole_kernel_factors"] = sorted(
        {int(factor)
         for factor in rejected_factors
         if int(factor) > 1})
    if forced_candidate in ("all_simd", "all_simt_only", "mixed_simd_simt"):
        reroute["backend_forced_route_candidate"] = forced_candidate
    return reroute


def _attach_backend_cross_route_feedback(metadata, feedback, adjusted_report=None,
                                         pre_backend_candidate="all_simt_only"):
    """Keep backend evidence in the report produced by the materialized reroute.

    A whole-SIMT candidate can be structurally legal but fail every backend
    factor compile because its generated private stack exceeds the hardware
    limit.  The compiler then reroutes the kernel through the all-SIMD path.
    The original report was written before that backend probe, so leaving it
    untouched makes the measured SIMD binary look like an unexplained
    low-cost SIMT prediction.  Rewrite the selected route fields together
    with the backend evidence so the report describes the binary that was
    actually materialized.
    """
    try:
        report_data = json.loads(metadata.get("auto_simt_scope_report", ""))
        adjusted_data = json.loads(adjusted_report) if adjusted_report else report_data
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return
    candidate_costs = report_data.get("candidate_costs")
    adjusted_costs = adjusted_data.get("candidate_costs", {})
    if isinstance(candidate_costs, dict) and isinstance(adjusted_costs, dict):
        if "all_simt_only" in adjusted_costs:
            candidate_costs["all_simt_only"] = adjusted_costs["all_simt_only"]
    selected_candidate = feedback.get("post_backend_selected_candidate",
                                      report_data.get("effective_decision_kind"))
    report_data["backend_resource_feedback"] = feedback
    report_data["pre_backend_decision_kind"] = pre_backend_candidate
    report_data["backend_recommended_decision_kind"] = selected_candidate
    report_data["decision_kind"] = selected_candidate
    report_data["effective_decision_kind"] = selected_candidate
    report_data["recommended_decision_kind"] = selected_candidate
    report_data["selection_source"] = "stage_cost_model_with_backend_resource_feedback"
    report_data["application_reason"] = "backend_resource_cross_route_reselection"

    # Select the elapsed prediction belonging to the rerouted candidate.  The
    # native report stores the candidate resource and elapsed totals for every
    # route, while ``selected_route_elapsed_prediction`` is a single selected
    # object.  Reconstruct its fixed boundary term from those paired totals so
    # downstream audits do not retain the pre-reroute SIMT prediction.
    elapsed_costs = report_data.get("candidate_elapsed_costs", {})
    resource_costs = report_data.get("candidate_costs", {})
    if (selected_candidate in elapsed_costs and selected_candidate in resource_costs):
        try:
            elapsed_total = float(elapsed_costs[selected_candidate])
            resource_total = float(resource_costs[selected_candidate])
            old_elapsed = report_data.get("selected_route_elapsed_prediction", {})
            prediction = dict(old_elapsed) if isinstance(old_elapsed, dict) else {}
            frequency = float(prediction.get("system_counter_frequency_mhz", 1000.0))
            fixed = max(0.0, elapsed_total - resource_total)
            prediction.update({
                "candidate": selected_candidate,
                "base_selection_score": resource_total,
                "pre_backend_base_selection_score": resource_total,
                "fixed_system_cycles": fixed,
                "predicted_system_counter_cycles": elapsed_total,
                "predicted_microseconds": elapsed_total / max(1.0, frequency),
                "reason": "backend_resource_cross_route_reselection",
            })
            if selected_candidate == "all_simd":
                prediction["superblock_factor"] = 1
            report_data["selected_route_elapsed_prediction"] = prediction
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    selected_route = ((report_data.get("stage_model") or {}).get("routes") or {}).get(selected_candidate)
    if isinstance(selected_route, dict):
        report_data["selected_superblock_factor"] = int(selected_route.get("route_superblock_factor", 1) or 1)
        report_data["effective_runtime_factor"] = int(
            selected_route.get("route_superblock_factor", 1) or 1)
        report_data["full_group_count"] = selected_route.get("runtime_full_group_count", 0)
        report_data["tail_count"] = selected_route.get("runtime_tail_program_count", 0)
        for key in ("runtime_loop_iteration_count", "runtime_physical_program_count",
                    "runtime_scheduling_slot_count", "runtime_logical_programs_per_scheduling_slot"):
            if key in selected_route:
                top_key = ("logical_programs_per_scheduling_slot"
                           if key == "runtime_logical_programs_per_scheduling_slot" else key)
                report_data[top_key] = selected_route[key]
    report_data["selection_source"] = "stage_cost_model_with_backend_resource_feedback"
    updated = json.dumps(report_data, sort_keys=True, separators=(",", ":"))
    metadata["auto_simt_scope_report"] = updated
    dump_path = metadata.get("auto_simt_scope_dump")
    if dump_path:
        _write_text_atomically(dump_path, updated)


def _default_auto_simt_profile_name(arch: str) -> str:
    """Select the hardware-specific StageCostModel profile by target arch.

    The route model is evaluated before lowering, but its calibration is still
    target-specific.  Falling back to the historical David-V100 profile for
    every target silently discards the 950PR/950DT layout and local-memory
    envelopes, so keep the generic profile only for unknown architectures.
    """
    normalized = (arch or "").lower().replace("-", "_")
    if "950pr" in normalized:
        return "simd_simt/ascend_950pr_simd_simt_v1.json"
    if "950dt" in normalized:
        return "simd_simt/ascend_950dt_simd_simt_v1.json"
    return "simd_simt/david_v100_simd_simt_v1.json"


def _apply_cpp_simd_simt_decision(metadata, effective: str, superblock_factor: int, report: str,
                                  base_num_warps: int = 1) -> None:
    """Translate the native decision into the backend execution contract."""
    metadata["auto_simt_scope_report"] = report
    metadata["auto_simt_superblock_factor"] = superblock_factor
    metadata["auto_simt_effective_kind"] = effective
    if effective == "all_simd":
        # ``simd_simt`` is the user's request to run the Route Model, not the
        # backend mode after selection.  Keeping it here makes later compiler
        # stages add mixed-runtime flags even though no SIMT scope exists.
        metadata["compile_mode"] = "simd"
        metadata["parallel_mode"] = "simd"
        # All-SIMD can still execute AutoBlockify V1 as a logical-program loop.
        # Keep the policy resolved before the cost model instead of disabling it.
        v1_materializable = bool(metadata.get("route_transform_v1_materializable", False))
        # Prefix scans with multiple result streams are lowered correctly by
        # the ordinary SIMD pipeline, while the AutoBlockify V1 loop changes
        # the scan carry partition and can corrupt the tail result.  Keep the
        # route-model decision (all-SIMD) but disable the executable V1 loop
        # for this structural operation class.
        try:
            report_data = json.loads(report or "{}")
            stages = (report_data.get("stage_model") or {}).get("logical_stages") or []
            has_prefix_scan = any(
                bool((stage.get("features") or {}).get("has_prefix_scan"))
                or bool((stage.get("workload") or {}).get("prefix_scan_instances_per_iteration"))
                for stage in stages
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            has_prefix_scan = False
        if has_prefix_scan:
            metadata["auto_blockify_v1_enabled"] = False
            metadata["auto_blockify_v1_runtime_cap"] = False
            metadata["auto_blockify_v1_disable_reasons"] = ["all_simd_prefix_scan"]
        else:
            metadata["auto_blockify_v1_enabled"] = v1_materializable
            metadata["auto_blockify_v1_runtime_cap"] = v1_materializable
    elif effective == "mixed_simd_simt":
        metadata["auto_simt_requested_kind"] = effective
        # Mixed execution still needs V1 at factor one: its persistent
        # logical-program loop distributes the launch grid across physical
        # cores and is independent of whether F2/F4 SuperBlock aggregation is
        # profitable.  F2/F4 remain legal candidates, but must not be used as
        # the admission condition for V1 itself.
        scope_factor = max(1, int(superblock_factor))
        v1_materializable = bool(metadata.get("route_transform_v1_materializable", False))
        metadata["auto_blockify_v1_enabled"] = v1_materializable
        metadata["auto_blockify_v1_runtime_cap"] = v1_materializable
        metadata["auto_simt_scope_superblock_factor"] = scope_factor
    elif effective == "all_simt_only":
        # Auto route selection owns this decision.  When the user did not
        # explicitly disable V1, make the selected pure-SIMT plan
        # executable instead of depending on a separate environment switch.
        if metadata.get("auto_simt_whole_kernel_superblock_materializable", False):
            metadata["auto_blockify_v1_enabled"] = True


def _skip_simd_simt_layout_merge(mod) -> bool:
    """Keep multi-result Triton scans on their native layout path.

    Layout coalescing rewrites the tensor shape feeding ``tt.scan``.  For a
    scan that returns multiple values, the resulting carry partition is not
    equivalent on the current A5 lowering and can corrupt the tail lanes.
    This is a structural guard and does not depend on a kernel or case name.
    """
    try:
        return "tt.scan" in str(mod)
    except Exception:
        return False


def _preserve_all_simd_auto_blockify_contract(metadata) -> None:
    """Keep the executable V1 loop that the all-SIMD candidate was scored on.

    The Route Model analyzes the post-AutoBlockify V1 clone.  Clearing this
    flag after it selects all-SIMD makes the final BiShengIR command omit
    ``--enable-auto-blockify-loop`` and materializes a different, slower
    binary than the candidate that won.  Preserve the already-resolved shared
    policy; this helper never enables V1 when analysis rejected it.
    """
    enabled = bool(metadata.get("auto_blockify_v1_enabled", False))
    metadata["auto_blockify_v1_enabled"] = enabled
    metadata["auto_blockify_v1_runtime_cap"] = enabled


def _selected_npuir_superblock_factor(metadata, opt) -> int:
    """Return the factor owned by the active route.

    CostModel-selected routes publish their factor in metadata. Explicit
    manual routes do not run the CostModel and therefore must retain the
    launch option. Reading ``auto_simt_scope_superblock_factor``
    unconditionally made an old/default F1 value shadow explicit F2/F4.
    """
    effective = metadata.get("auto_simt_effective_kind")
    if (effective in ("all_simd", "mixed_simd_simt")
            or (effective is None and metadata.get("compile_mode") == "simd_simt")):
        # The process-wide factor owns the surrounding SIMD/AIC scheduling
        # graph and must stay F1. The selected F2/F4 value is carried on the
        # scope itself and restored only in its outlined SIMT module.
        selected = 1
    elif effective == "all_simt_only":
        selected = metadata.get("auto_simt_superblock_factor", opt.superblock_factor)
    else:
        selected = opt.superblock_factor
    return max(1, int(selected or 1))


def _append_pure_simt_auto_blockify_options(options, metadata, opt) -> None:
    """Pass the executable V1 schedule contract to bishengir-compile."""
    if (not metadata.get("auto_blockify_v1_enabled", False) or not metadata.get("auto_blockify_v1_runtime_cap", False)):
        return
    if not opt.enable_ta_auto_blockify_v1:
        options.append("--enable-auto-blockify-loop")
    selected_factor = _selected_npuir_superblock_factor(metadata, opt)
    # F1 is the compiler default.  Older 950PR BiShengIR packages expose the
    # blockify pass but not the optional SuperBlock-factor flag, so emitting
    # ``--super-block-factor=1`` makes otherwise ordinary SIMT compilation
    # fail before IR lowering starts.
    if selected_factor > 1:
        options.append(f"--super-block-factor={selected_factor}")


def _can_materialize_scope_superblock(metadata, opt, whole_kernel_materializable: bool) -> bool:
    """Whether the mixed backend can execute a selected F2/F4 plan.

    The selected Stage is materialized as ``scope.scope<vector_mode=simt>``.
    NPUIR batches F logical programs around that scope after bufferization,
    while the process-wide V1 factor remains one.
    """
    del metadata
    return bool(whole_kernel_materializable and getattr(opt, "compile_on_910_95", False)
                and int(getattr(opt, "num_warps", 0) or 0) > 0)


def _publish_route_transform_capability(metadata, opt) -> str:
    """Publish the single capability fact consumed by scoring and lowering.

    Layout transforms and the V1 policy have already run when this function is
    called.  The returned JSON is passed unchanged to the native Route Model so
    the report, candidate legality, and backend materialization describe the
    same transformed TTIR.
    """
    target_supported = bool(getattr(opt, "compile_on_910_95", False))
    num_warps = max(1, int(getattr(opt, "num_warps", 1) or 1))
    v1_enabled = bool(metadata.get("auto_blockify_v1_enabled", False))
    v1_materializable = target_supported and v1_enabled

    disable_reasons = metadata.get("auto_blockify_v1_disable_reasons", [])
    if not isinstance(disable_reasons, list):
        disable_reasons = [str(disable_reasons)]
    if not v1_enabled and not disable_reasons:
        disable_reasons = ["not_requested_or_explicitly_disabled"]
    if not target_supported:
        disable_reasons.append("target_does_not_support_auto_blockify_v1")

    modeled_factors = _get_modeled_superblock_factors()
    current_scope_factors = _get_current_scope_superblock_factors()
    legal_factors = [factor for factor in modeled_factors if num_warps * factor <= 64]
    if not legal_factors:
        legal_factors = [1]

    rejected_whole_kernel_factors = {
        int(factor)
        for factor in metadata.get("backend_rejected_whole_kernel_factors", [])
        if str(factor).isdigit()
    }
    whole_kernel_superblock_factors = (
        [factor for factor in legal_factors
         if factor not in rejected_whole_kernel_factors] if v1_materializable else [1]) or [1]
    superblock_factors = ([factor for factor in legal_factors
                           if factor in current_scope_factors] if v1_materializable else [1]) or [1]
    coalesce_factor = max(1, int(metadata.get("ttir_layout_coalesce_factor", 1) or 1))

    coalesce_axis = metadata.get("ttir_layout_coalesce_axis", -1)
    if not isinstance(coalesce_axis, int):
        coalesce_axis = -1
    physical_vector_cores = max(0, int(getattr(opt, "physical_vector_core_count_hint", 0) or 0))
    physical_aic_cores = max(0, int(getattr(opt, "physical_aicore_count_hint", 0) or 0))
    if not physical_vector_cores or not physical_aic_cores:
        try:
            npu_utils = NPUUtils()
            if not physical_vector_cores:
                physical_vector_cores = int(npu_utils.get_aivector_core_num())
            if not physical_aic_cores:
                physical_aic_cores = int(npu_utils.get_aicore_num())
        except Exception:
            pass
    if not physical_vector_cores:
        try:
            physical_vector_cores = int(NPUUtils().get_aivector_core_num())
        except Exception:
            physical_vector_cores = 0
    source_logical_program_count = max(0, int(getattr(opt, "logical_program_count_hint", 0) or 0))
    transformed_logical_program_count = ((source_logical_program_count + coalesce_factor - 1) //
                                         coalesce_factor if source_logical_program_count else 0)
    capability = {
        "schema_version":
        2,
        "automatic_route_selection":
        getattr(opt, "auto_simt_scope_mode", "off") == "auto",
        "layout_merge_applied":
        bool(metadata.get("ttir_layout_merge_applied", False)),
        "layout_coalescing_applied":
        coalesce_factor > 1,
        "layout_coalescing_factor":
        coalesce_factor,
        "layout_coalescing_axis":
        coalesce_axis,
        # Compatibility aliases for the pre-layout terminology consumed by
        # older Python guards.  Both names describe the same resolved fact;
        # neither is an independent route decision.
        "row_coalescing_applied":
        coalesce_factor > 1,
        "row_coalescing_factor":
        coalesce_factor,
        "row_coalescing_axis":
        coalesce_axis,
        "auto_blockify_v1_requested":
        bool(metadata.get("auto_blockify_v1_requested", False)),
        "auto_blockify_v1_materializable":
        v1_materializable,
        "auto_blockify_v1_disable_reasons":
        sorted(set(disable_reasons)),
        "whole_kernel_superblock_factors":
        whole_kernel_superblock_factors,
        "scope_superblock_factors":
        superblock_factors,
        "source_logical_program_count_hint":
        source_logical_program_count,
        "logical_program_count_hint":
        transformed_logical_program_count,
        # Legacy readers still consume these two derived fields.  Keep them
        # as compatibility views of the facts above; route scheduling remains
        # owned by the native RouteSolver.
        "modeled_superblock_factors":
        modeled_factors,
        "superblock_runtime_groups": {
            str(factor): {
                "full_group_count": transformed_logical_program_count // factor,
                "tail_count": transformed_logical_program_count % factor,
            }
            for factor in modeled_factors
        },
        "physical_vector_core_count_hint":
        physical_vector_cores,
        "physical_aicore_count_hint":
        physical_aic_cores,
        "runtime_scalar_values": [{
            "argument_index": int(argument_index),
            "value": int(value),
        } for argument_index, value in getattr(opt, "auto_simt_runtime_scalar_values", ())],
        "cv_aiv_subblock_count": (2 if physical_aic_cores > 0 and physical_vector_cores == 2 * physical_aic_cores
                                  and getattr(opt, "enable_auto_bind_sub_block", None) is not False
                                  and getattr(opt, "auto_tile_and_bind_subblock", True) else 1),
    }
    capability["dynamic_cv_pipeline_requested"] = bool(getattr(opt, "enable_dynamic_cv_pipeline", False))
    forced_candidate = metadata.get("backend_forced_route_candidate")
    if forced_candidate in ("all_simd", "all_simt_only", "mixed_simd_simt"):
        capability["backend_forced_route_candidate"] = forced_candidate
    # Keep this contract factual. Route-specific Q/group/tail composition has
    # one implementation in the native RouteSolver; duplicating that formula
    # in Python previously let the two sides drift apart.
    capability_json = json.dumps(capability, sort_keys=True, separators=(",", ":"))
    metadata["route_transform_capability"] = capability_json
    metadata["route_transform_v1_materializable"] = v1_materializable
    metadata["route_transform_whole_kernel_factors"] = ",".join(map(str, whole_kernel_superblock_factors))
    metadata["route_transform_scope_factors"] = ",".join(map(str, superblock_factors))
    return capability_json


def _run_cpp_simd_simt_selection_pass(mod, selection_mode, profile, opt,
                                      whole_kernel_materializable, scope_materializable,
                                      whole_kernel_factors, scope_factors, logical_program_count,
                                      analysis_path, capability_json, report_path):
    """Run one native route-selection/materialization pass."""
    selection_pm = ir.pass_manager(mod.context)
    selection_pm.enable_debug()
    ascend.passes.ttir.add_select_simd_simt_costmodel(
        selection_pm,
        selection_mode,
        profile,
        str(opt.arch),
        int(opt.num_warps),
        bool(opt.compile_on_910_95),
        whole_kernel_materializable,
        scope_materializable,
        max(whole_kernel_factors, default=1),
        max(scope_factors, default=1),
        int(logical_program_count),
        analysis_path,
        capability_json,
        report_path,
    )
    ascend.passes.ttir.add_materialize_simt_scopes(selection_pm)
    selection_pm.run(mod, "select_simd_simt_costmodel")


def _force_auto_pure_route_candidate(probe_report, capability_json):
    """Restrict legacy Auto reports to the cheapest whole-kernel candidate."""
    try:
        # A backend resource probe may have already selected a concrete
        # whole-kernel reroute.  Preserve that decision across the second
        # native pass; re-running the legacy pure-route tie breaker here can
        # undo the backend evidence and materialize a different route.
        capability = dict(json.loads(capability_json))
        forced_candidate = capability.get("backend_forced_route_candidate")
        if forced_candidate in ("all_simd", "all_simt_only"):
            return capability_json
        probe_data = json.loads(probe_report)
        selectable = set(probe_data.get("selectable_candidates", []))
        costs = probe_data.get("candidate_elapsed_costs", {})
        pure_candidates = [candidate for candidate in ("all_simd", "all_simt_only")
                           if candidate in selectable and isinstance(costs.get(candidate), (int, float))]
        if pure_candidates:
            forced = min(pure_candidates, key=lambda candidate: (float(costs[candidate]), candidate))
            capability["backend_forced_route_candidate"] = forced
            return json.dumps(capability, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return capability_json


def _run_cpp_simd_simt_costmodel(mod, metadata, opt, analysis_ttir_code: str = "") -> str:
    """Run native selection/materialization; Python only schedules the passes."""
    mode = opt.auto_simt_scope_mode
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":
        return "backend_default"

    profile = opt.auto_simt_model_profile or str(_costmodel_profiles_dir() / _default_auto_simt_profile_name(opt.arch))
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    capability_json = metadata.get("route_transform_capability")
    if not capability_json:
        capability_json = _publish_route_transform_capability(metadata, opt)
    capability = json.loads(capability_json)
    whole_kernel_factors = capability.get("whole_kernel_superblock_factors", [1])
    scope_factors = capability.get("scope_superblock_factors", [1])
    whole_kernel_superblock_materializable = bool(metadata.get("route_transform_v1_materializable", False))
    metadata["auto_simt_whole_kernel_superblock_materializable"] = (whole_kernel_superblock_materializable)
    # Mixed F2/F4 requires an outer factor-one V1 loop that NPUIR can refine
    # around the selected local scope.
    scope_superblock_materializable = _can_materialize_scope_superblock(metadata, opt,
                                                                        whole_kernel_superblock_materializable)
    metadata["auto_simt_scope_superblock_materializable"] = scope_superblock_materializable
    with tempfile.TemporaryDirectory() as tmpdir:
        analysis_path = ""
        if analysis_ttir_code:
            analysis_path = os.path.join(tmpdir, "post_auto_blockify_v1.ttir.mlir")
            Path(analysis_path).write_text(analysis_ttir_code, encoding="utf-8", newline="\n")

        # Auto is intentionally a whole-kernel choice.  The native pass in
        # older installed extensions still exposes mixed as the cheapest
        # candidate, so probe its immutable report first and force the best
        # legal all-SIMD/all-SIMT candidate for the real materialization pass.
        if mode == "auto":
            _run_cpp_simd_simt_selection_pass(
                mod, "report", profile, opt, whole_kernel_superblock_materializable,
                scope_superblock_materializable, whole_kernel_factors, scope_factors,
                capability.get("logical_program_count_hint", 0), analysis_path, capability_json, "")
            probe_report = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.report_json")
            capability_json = _force_auto_pure_route_candidate(probe_report, capability_json)

        _run_cpp_simd_simt_selection_pass(
            mod, mode, profile, opt, whole_kernel_superblock_materializable,
            scope_superblock_materializable, whole_kernel_factors, scope_factors,
            capability.get("logical_program_count_hint", 0), analysis_path,
            capability_json, str(opt.auto_simt_scope_dump))

    report = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.report_json")
    effective = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.effective")
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    superblock_factor = (get_int_attr(mod, "ascend.simt_costmodel.superblock_factor") if get_int_attr else 1)
    if not isinstance(superblock_factor, int) or superblock_factor < 1:
        superblock_factor = 1
    if not report or effective not in {"all_simd", "all_simt_only", "mixed_simd_simt", "backend_default"}:
        raise RuntimeError("invalid native SIMD/SIMT costmodel result")
    _apply_cpp_simd_simt_decision(metadata, effective, superblock_factor, report, int(opt.num_warps))
    # Auto is restricted to whole-kernel routes.  Keep this fact available to
    # the backend resource-feedback stage so an older native extension cannot
    # request a mixed cross-route rerun after the pure route was selected.
    metadata["auto_simt_disallow_mixed"] = mode == "auto"
    return effective


