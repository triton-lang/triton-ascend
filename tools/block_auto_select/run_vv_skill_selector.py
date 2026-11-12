"""CPU-only VV selector for prepared compilation artifacts; never launches NPU.

Use one grid-selected axis for each of factors 2/4, recheck resource proofs,
then rank with the instruction/dependency/DMA model. Unknown model coverage
falls back to the verified original artifact, with an explicit reason.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from backend_cost_model import Unsupported, read_cost_table
from cpu_resource_gate import sha256
from run_vv_contention_experiment import read, write
from vv_resource_gate import verify_vv_artifacts
from vv_candidate_plan import plan_candidates
from v38_runtime_policy import rejection as runtime_rejection
from vv_contention_model import choose
from vv_skill_model import DATA, VERSION, InstructionModel, analyze_candidate

REJECTED = {"UB_REJECTED", "COMPILE_REJECTED", "FRONTEND_REJECTED", "RUNTIME_REJECTED"}


def select(candidates, *, experimental_mixed_ranking=False):
    scored = {
        k: v
        for k, v in candidates.items()
        if v["status"] == "STATIC_COST_ESTIMATED" and v.get("resource_checked") is True and type(v.get("score")) in (
            int, float) and math.isfinite(v["score"]) and v["score"] >= 0
    }
    complete = bool(candidates) and all(k in scored or v["status"] in REJECTED for k, v in candidates.items())
    best = choose(scored)
    baseline = candidates.get("unmerged", {})
    experimental = any(v.get('experimental_mixed_model') for v in scored.values())
    if complete and experimental and not experimental_mixed_ranking:
        baseline_ok = baseline.get('resource_checked') is True and bool(baseline.get('binary'))
        return dict(
            selected='unmerged' if baseline_ok else None, selection_reason='BASELINE_FALLBACK_EXPERIMENTAL_MIX'
            if baseline_ok else 'NO_CONFIRMED_BASELINE_FOR_EXPERIMENTAL_MIX', model_complete=False,
            analysis_complete=True, static_best=best)
    if complete and best is not None:
        return dict(selected=best, selection_reason="LOWEST_STATIC_SCORE", model_complete=True, analysis_complete=True,
                    static_best=best)
    if baseline.get("resource_checked") is True and baseline.get("binary"):
        return dict(selected="unmerged", selection_reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", model_complete=False,
                    analysis_complete=complete)
    return dict(selected=None, selection_reason="NO_CONFIRMED_BASELINE_OR_COMPLETE_RANKING", model_complete=False,
                analysis_complete=complete)


def install_cpu_guard(manifest_path, manifest, cost_table, output):
    """Limit analysis to declared artifacts, model code and Python libraries."""
    allowed = {Path(manifest_path).resolve(), Path(cost_table).resolve()}
    for case in manifest["cases"]:
        allowed.add(Path(case["fixture"]).resolve())
        for candidate in case["candidates"].values():
            for record in candidate.get("files", {}).values():
                allowed.add(Path(record["path"]).resolve())
    code = Path(__file__).resolve().parent
    libraries = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    output = output.resolve()
    events = {"denied": [], "timing_labels_read": False, "device_executed": False}

    def audit(event, args):
        reason = None
        if event == "import" and args[0].split(".")[0] in ("torch", "torch_npu", "triton", "acl", "pyacl"):
            reason = "device runtime import"
        elif event in ("subprocess.Popen", "os.system", "socket.connect"):
            reason = event
        elif event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1]
            writing = (isinstance(mode, str) and any(c in mode
                                                     for c in "wax+")) or (isinstance(args[2], int) and args[2] &
                                                                           (os.O_WRONLY | os.O_RDWR))
            if writing:
                if not path.is_relative_to(output):
                    reason = "write outside new result directory: " + str(path)
            elif not (path in allowed or path.is_relative_to(code) or path.is_relative_to(output)
                      or any(path.is_relative_to(root) for root in libraries)):
                reason = "undeclared analysis input: " + str(path)
            if str(path).startswith(("/dev/davinci", "/dev/devmm")):
                reason = "device access"
        if reason:
            events["denied"].append(reason)
            raise RuntimeError(reason)

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidth", choices=("ini", "legacy"), default="ini")
    parser.add_argument('--experimental-mixed-ranking', action='store_true',
                        help='Allow the uncalibrated AIV MIX score to determine selection')
    parser.add_argument("--runtime-core-floor", type=int)
    parser.add_argument("--require-optimized-evidence", action="store_true")
    parser.add_argument('--lowered-entry', action='store_true')
    args = parser.parse_args()
    if args.runtime_core_floor is not None and args.runtime_core_floor < 0:
        parser.error("runtime core floor must be nonnegative")
    manifest = read(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    audit = install_cpu_guard(args.manifest, manifest, args.cost_table, args.output)
    try:
        model = InstructionModel()
        table = read_cost_table(args.cost_table.read_text())
        if not table:
            raise ValueError("empty target cost table")
        code = Path(__file__).parent
        sources = [
            Path(__file__), code / "backend_cost_model.py", code / "cpu_resource_gate.py",
            code / "run_vv_contention_experiment.py", code / "vv_contention_model.py", code / "vv_candidate_plan.py",
            code / "vv_skill_model.py", code / "vv_resource_gate.py", code / "vv_library_model.py",
            code / "vv_mixed_model.py", code / "vv_optimized_ir.py", code / "vv_ready_schedule.py",
            code / "v38_runtime_policy.py", code / "vv_simt_helpers.py", code / "vv_lowered_entry.py"
        ]
        sources += sorted(p for p in (code / "vv_library_data").iterdir() if p.is_file())
        sources += sorted(p for p in (code / "vv_mixed_data").iterdir() if p.is_file())
        sources += sorted(p for p in DATA.iterdir() if p.is_file())
        provenance = read(DATA / "provenance.json")
        for name, digest in provenance["sha256"].items():
            if sha256(DATA / name) != digest:
                raise ValueError("frozen skill asset changed: " + name)
        protocol = dict(model=VERSION, bandwidth=args.bandwidth, candidate_limit=3,
                        runtime_core_floor=args.runtime_core_floor,
                        require_optimized_evidence=args.require_optimized_evidence, lowered_entry=args.lowered_entry,
                        experimental_mixed_ranking=args.experimental_mixed_ranking, factor_scope=[1, 2, 4],
                        device_executed=False, compilation_executed=False, timing_labels_read=False,
                        integrated_into_jit=False, purpose="CPU selector; existing-artifact development replay",
                        parameters="instruction models plus uncalibrated dependency/scalar/DMA costs",
                        sources={str(p): sha256(p)
                                 for p in sources}, manifest=dict(path=str(args.manifest),
                                                                  sha256=sha256(args.manifest)),
                        cost_table=dict(path=str(args.cost_table), sha256=sha256(args.cost_table)))
        write(args.output / "protocol.json", protocol)
        rows, ids = [], set()
        for case in manifest["cases"]:
            ident = case["id"]
            if ident in ids or not ident or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in ident):
                raise ValueError("invalid or duplicate case ID")
            ids.add(ident)
            fixture = read(case["fixture"])
            if fixture["contract"] != case["contract"]:
                raise ValueError("manifest differs from fixture launch contract")
            plan = plan_candidates(fixture["contract"]["grid"], case["candidates"])
            row = dict(id=ident, name=case["name"], plan=plan, candidates={},
                       fixture=dict(path=case["fixture"], sha256=sha256(case["fixture"])))
            family = [case['candidates'][k] for k in plan['keys'] if case['candidates'][k].get('resource_checked')]
            optimized = any('optimized_proof' in c['files'] for c in family)
            family_complete = all('optimized_proof' in c['files'] for c in family)
            for key in plan["keys"]:
                candidate = case["candidates"][key]
                started = time.monotonic()
                value = dict(status=candidate["status"], resource_checked=False, axis=candidate["axis"],
                             factor=candidate["factor"], error=candidate.get("error"))
                if candidate.get("resource_checked") is True:
                    try:
                        files, final_ir, metadata, usage = verify_vv_artifacts(candidate)
                        value.update(resource_checked=True, binary=files["binary"], resource_proof=files["resource"],
                                     ub_bytes=usage["address_extent_bits"] / 8,
                                     ub_capacity_bytes=usage["capacity_bits"] / 8)
                    except (ValueError, KeyError, OSError) as exc:
                        value.update(status="RESOURCE_UNCONFIRMED", error=str(exc))
                    else:
                        refusal = (runtime_rejection(fixture['contract']['grid'], candidate['axis'],
                                                     candidate['factor'], args.runtime_core_floor, metadata)
                                   if args.runtime_core_floor is not None else None)
                        if refusal:
                            value.update(status='RUNTIME_REJECTED', error=refusal,
                                         runtime_core_floor=args.runtime_core_floor)
                        else:
                            try:
                                if args.require_optimized_evidence and 'optimized_proof' not in files:
                                    raise Unsupported('uniform new model requires optimized executable proof')
                                if optimized and not family_complete:
                                    raise Unsupported('optimized evidence must cover the complete candidate family')
                                value.update(
                                    analyze_candidate(
                                        Path(files["llvm"]["path"]).read_text(),
                                        Path(files["ttir"]["path"]).read_text(), final_ir, metadata,
                                        fixture["contract"], candidate["axis"], candidate["factor"], table, model=model,
                                        bandwidth=args.bandwidth, library_evidence=files.get("libdevice"),
                                        mixed_evidence=files, optimized_evidence=files if optimized else None,
                                        lowered_entry=args.lowered_entry))
                            except (ValueError, KeyError, OSError) as exc:
                                value.update(status="STATIC_ANALYSIS_UNSUPPORTED", error=str(exc))
                value["analysis_seconds"] = time.monotonic() - started
                row["candidates"][key] = value
                write(args.output / "scores" / ident / (key + ".json"), value)
                print(ident, key, value["status"], value.get("score", value.get("error")), flush=True)
            row.update(select(row["candidates"], experimental_mixed_ranking=args.experimental_mixed_ranking))
            row["all_candidates_compared"] = row["model_complete"] and not plan["omitted"]
            selected = row["candidates"].get(row["selected"], {})
            row["selected_artifact"] = {
                k: selected[k]
                for k in ("binary", "resource_proof", "axis", "factor")
                if k in selected
            }
            rows.append(row)
            write(args.output / "static-predictions.json", rows)
            print(ident, "SELECTED", row["selected"], row["selection_reason"], flush=True)
        write(
            args.output / "static-complete.json",
            dict(cases=len(rows), predictions_sha256=sha256(args.output / "static-predictions.json"),
                 protocol_sha256=sha256(args.output / "protocol.json"), completed_unix=time.time(),
                 timing_labels_read=False, device_executed=False))
    finally:
        write(args.output / "cpu-access-audit.json", audit)


if __name__ == "__main__":
    main()
