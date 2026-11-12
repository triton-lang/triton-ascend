"""Prepare hash-bound library/default evidence and retry CPU resource failures."""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import cpu_resource_gate as gate
    from run_vv_contention_experiment import read, write
    from vv_resource_gate import inspect_vv_ub, verify_vv_artifacts
    from vv_library_model import DATA

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read(args.manifest)
    library = read(DATA / "provenance.json")
    records = []

    def record(path):
        return dict(path=str(path), sha256=gate.sha256(path))

    for case in manifest["cases"]:
        for key, candidate in case["candidates"].items():
            if candidate.get("status") == "RESOURCE_UNCONFIRMED" and candidate.get("front"):
                front = Path(candidate["front"])
                old_backend = front.parent.parent / "backend" / candidate["name"]
                request = read(old_backend / "request.json")
                if gate.sha256(request["metadata"]["path"]) != request["metadata"]["sha256"]:
                    raise ValueError("runtime metadata changed")
                metadata = read(request["metadata"]["path"])
                if metadata["parallel_mode"] != "simd":
                    raise ValueError("retry compiler currently accepts SIMD input only")
                output = args.output / "backend" / candidate["name"]
                proof = gate.compile_candidate(request, output)
                if proof["returncode"] != 0:
                    candidate.update(status=proof["status"], resource_checked=False, error=proof.get("reason"))
                    records.append(dict(id=case["id"], key=key, status=proof["status"]))
                    continue
                final = output / "temps/module.hivm.opt.mlir"
                binary = output / "kernel.o"
                if (binary.read_bytes()[:4] != b"\x7fELF"
                        or gate.PASS_MARKER not in (output / "stderr.log").read_text()):
                    raise ValueError("fresh backend allocation proof missing")
                paths = dict(metadata=Path(request["metadata"]["path"]), ttir=front / "candidate.ttir",
                             ttadapter=front / "candidate.ttadapter", final_ir=final, binary=binary,
                             llvm=output / "temps/kernel.ll", compiler=Path(request["compiler"]))
                files = {name: record(p) for name, p in paths.items()}
                usage = inspect_vv_ub(final.read_text(), request["target"], metadata, proof, files)
                proof.update(status="UB_RESOURCE_CHECKED", resource_checked=True, ub=usage, artifact=files["binary"],
                             final_ir_sha256=files["final_ir"]["sha256"],
                             proof="fresh CPU compile and aligned conservative UB range recheck")
                write(output / "resource-result.json", proof)
                files["resource"] = record(output / "resource-result.json")
                candidate.update(status=proof["status"], resource_checked=True, files=files, error=None)
                records.append(
                    dict(id=case["id"], key=key, status=proof["status"], binary=files["binary"],
                         ub_bytes=usage["address_extent_bits"] / 8,
                         old_binary_sha256=gate.sha256(old_backend / "kernel.o")))
            if candidate.get("resource_checked") is not True:
                continue
            files = candidate["files"]
            proof = read(files["resource"]["path"])
            metadata = read(files["metadata"]["path"])
            if metadata["parallel_mode"] == "mix_simd_simt":
                from vv_mixed_model import attach_mixed_evidence
                attach_mixed_evidence(candidate)
                defaults = Path(files["resource"]["path"]).parents[2] / "backend-default-options.txt"
                item = record(defaults)
                if item["sha256"] != proof["default_option_evidence_sha256"]:
                    raise ValueError("compiler default evidence changed")
                files["backend_defaults"] = item
            if "@pow_fp32(" in Path(files["llvm"]["path"]).read_text():
                lib = Path(files["compiler"]["path"]).parent.parent.parent / "lib/libdevice.10.bc"
                item = record(lib)
                if item["sha256"] != library["libdevice_sha256"]:
                    raise ValueError("candidate libdevice differs from pinned analysis body")
                files["libdevice"] = item
            try:
                verify_vv_artifacts(candidate)
            except ValueError as exc:
                candidate.update(resource_checked=False, status="RESOURCE_UNCONFIRMED", error=str(exc))
            print(case["id"], key, candidate["status"], candidate.get("error"), flush=True)
    write(args.output / "manifest.json", manifest)
    write(args.output / "preparation.json",
          dict(source_manifest=record(args.manifest), records=records, device_executed=False, history_read=False))


if __name__ == "__main__":
    main()
