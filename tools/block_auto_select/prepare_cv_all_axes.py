"""Compile every eligible CV grid-axis candidate, retaining individual failures.

Inputs are explicit frozen specifications and a pinned compiler installation.
Identical existing artifacts can be reused after source, specialization, native
compiler, backend and UB-proof identity checks. No measurements are read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path
import subprocess
import traceback

from cpu_resource_gate import compile_candidate, sha256
from cv_candidate_plan import VERSION, plan_candidates
from artifact_io import read, write
from run_cv_spec_experiment import checked_candidate


def record(path):
    return dict(path=str(path), sha256=sha256(path))


def reusable(item, expected_request, compiler_hash):
    if "files" not in item or item["files"]["compiler"]["sha256"] != compiler_hash:
        return False
    front_path = Path(item["files"]["metadata"]["path"]).with_name("result.json")
    front = read(front_path)
    if front["status"] != "TTADAPTER_COMPILED" or front.get("forbidden_attempts") or front.get(
            "device_runtime_modules"):
        return False
    request = front["request"]
    for field in ("case_sha256", "kernel_sha256", "kernel_name", "native_sha256", "target", "options"):
        if request[field] != expected_request[field]:
            return False
    checked_candidate(item)
    return record(front_path)


def prepare_one(job, output, python, toolchain):
    front = output / "frontend" / job["name"]
    back = output / "backend" / job["name"]
    log = output / "logs" / (job["name"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    result = dict(axis=job["axis"], factor=job["factor"], resource_checked=False, device_executed=False,
                  status="PREPARATION_ERROR")
    try:
        with log.open("w") as stream:
            subprocess.run([
                python, "-I", "-S",
                str(Path(__file__).with_name("cpu_cv_frontend.py")), "--request", job["request"], "--output",
                str(front)
            ], stdout=stream, stderr=subprocess.STDOUT, timeout=180, check=False)
        frontend = read(front / "result.json")
        if frontend["forbidden_attempts"] or frontend["device_runtime_modules"]:
            raise RuntimeError("CPU frontend boundary violation")
        if frontend["status"] != "TTADAPTER_COMPILED":
            return dict(result, status="FRONTEND_REJECTED", error=frontend.get("error"),
                        rejection_evidence=record(front / "result.json"))
        with log.open("a") as stream:
            subprocess.run([
                python, "-I", "-S",
                str(Path(__file__).with_name("cpu_cv_backend_request.py")),
                str(front),
                str(toolchain)
            ], stdout=stream, stderr=subprocess.STDOUT, timeout=60, check=True)
        backend_request = read(front / "backend-request-finalized.json")
        backend = compile_candidate(backend_request, back, timeout=300)
        if backend["resource_checked"] is not True:
            return dict(result, status=backend["status"], error=backend.get("reason", backend.get("overflow")),
                        rejection_evidence=record(back / "result.json"))
        paths = dict(resource=back / "result.json", metadata=front / "backend-metadata.json",
                     ttir=front / "candidate.ttir", ttadapter=front / "candidate.ttadapter",
                     final_ir=back / "temps/module.hivm.opt.mlir", binary=back / "kernel.o",
                     compiler=Path(backend_request["compiler"]))
        for role in ("aic", "aiv"):
            path = back / "temps" / f"kernel_mix_{role}.ll"
            if path.exists():
                paths["llvm_" + role] = path
        single = back / "temps/kernel.ll"
        if single.exists() and "llvm_aic" not in paths:
            if '"cube-only-kernel"' not in single.read_text():
                raise ValueError("single CV LLVM module is not explicitly Cube")
            paths["llvm_aic"] = single
        result.update(status="UB_RESOURCE_CHECKED", resource_checked=True,
                      files={role: record(path)
                             for role, path in paths.items()}, source_frontend=record(front / "result.json"))
        checked_candidate(result)
    except Exception as error:
        result.update(status="PREPARATION_ERROR", resource_checked=False, error=str(error),
                      traceback=traceback.format_exc())
        result.pop("files", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specifications", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = args.toolchain / "site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile"
    compiler_hash = sha256(compiler)
    native_hash = sha256(args.toolchain / "site-packages/triton/_C/libtriton.so")
    reuse = {}
    for path in args.reuse_manifest:
        for case in read(path)["cases"]:
            for key, candidate in case["candidates"].items():
                reuse.setdefault((case["id"], key), []).append((path, candidate))
    specifications = read(args.specifications)
    cases, jobs, reuse_failures = [], [], []
    for source in specifications["cases"]:
        request = source["request"]
        if request["native_sha256"] != native_hash:
            raise ValueError("specification uses a different native compiler")
        for field in ("case", "kernel"):
            if sha256(request[field]) != request[field + "_sha256"]:
                raise ValueError("input changed: " + field)
        case = {key: copy.deepcopy(source[key]) for key in ("id", "name", "contract")}
        case["specification"] = record(request["case"])
        case["candidates"] = plan_candidates(case["contract"]["grid"])
        cases.append(case)
        for key, candidate in list(case["candidates"].items()):
            name = case["id"] + "-" + key
            if candidate["status"] == "NO_MERGE_OPPORTUNITY":
                path = args.output / "eligibility" / (name + ".json")
                write(path, dict(candidate, original_grid=case["contract"]["grid"], device_executed=False))
                candidate["rejection_evidence"] = record(path)
                continue
            expected = copy.deepcopy(request)
            expected["options"].update(block_batching_factor=candidate["factor"], block_batching_axis=candidate["axis"])
            match = None
            for reuse_path, old in reuse.get((case["id"], key), []):
                try:
                    front = reusable(old, expected, compiler_hash)
                    if front:
                        match = copy.deepcopy(old)
                        match.update(reused_from_manifest=record(reuse_path), source_frontend=front)
                        break
                except (ValueError, KeyError, OSError) as error:
                    reuse_failures.append(dict(case=case["id"], candidate=key, error=str(error)))
            if match:
                case["candidates"][key] = match
            else:
                path = args.output / "requests" / (name + ".json")
                write(path, expected)
                jobs.append(
                    dict(name=name, case=case["id"], key=key, axis=candidate["axis"], factor=candidate["factor"],
                         request=str(path)))
    manifest = dict(candidate_policy=VERSION, cases=cases)
    write(
        args.output / "protocol.json",
        dict(
            candidate_policy=VERSION, factors=[0, 2, 4], compiler_sha256=compiler_hash, native_sha256=native_hash,
            specifications=record(args.specifications), reuse_inputs=[record(p) for p in args.reuse_manifest],
            reuse_failures=reuse_failures, sources={
                name: sha256(Path(__file__).with_name(name))
                for name in ("prepare_cv_all_axes.py", "cv_candidate_plan.py", "cpu_cv_frontend.py",
                             "cpu_resource_gate.py", "cpu_cv_backend_request.py")
            }, device_executed=False, timings_used=False))
    write(args.output / "requested_manifest.json", manifest)
    write(args.output / "jobs.json", jobs)
    print("New CPU compilations:", len(jobs), flush=True)
    prepared = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, job, args.output, args.python, args.toolchain): job for job in jobs}
        for future in as_completed(futures):
            job, result = futures[future], future.result()
            next(c for c in cases if c["id"] == job["case"])["candidates"][job["key"]] = result
            prepared.append(dict(name=job["name"], **result))
            write(args.output / "prepared.json", prepared)
            write(args.output / "manifest.json", manifest)
            write(args.output / "progress.json", dict(completed=len(prepared), total=len(jobs)))
            print(job["name"], result["status"], result.get("error", ""), flush=True)
    write(args.output / "manifest.json", manifest)
    write(
        args.output / "complete.json",
        dict(
            manifest_sha256=sha256(args.output / "manifest.json"), cases=len(cases), new_compilations=len(jobs),
            device_executed=False, preparation_errors=sum(c["status"] == "PREPARATION_ERROR"
                                                          for case in cases
                                                          for c in case["candidates"].values())))


if __name__ == "__main__":
    main()
