"""Run the complete upstream community scope, retaining failures and interruptions.

This entry point is opt-in and does not add known failing tests to normal CI.
It does not import torch, triton or torch_npu in the supervising process.
"""

import argparse
import ast
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
BAD_OUTCOMES = {"failed", "error", "xpassed", "timeout", "crash"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(repository):
    paths = sorted((repository / "python/test/unit").rglob("*.py"))
    paths.append(repository / "python/test/conftest.py")
    return {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def check_scope(repository, manifest):
    files = sorted((repository / "python/test/unit").rglob("test*.py"))
    found = []
    for path in files:
        prefix = str(path.relative_to(repository)) + "::"
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                found.append(prefix + node.name)
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                found.extend(
                    prefix + node.name + "::" + item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"))
    expected = [item["id"] for item in manifest["definitions"]]
    if len(expected) != len(set(expected)) or len(found) != len(set(found)):
        raise ValueError("Duplicate definition identifiers")
    missing, added = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
    if missing or added:
        raise ValueError(f"Update the full manifest: missing={missing}, added={added}")
    actual_files = [str(path.relative_to(repository)) for path in files]
    if actual_files != manifest["files"]:
        raise ValueError("Test files differ from the full manifest")
    return {
        "files": len(files), "definitions": len(found), "class_methods": sum(item.count("::") == 2 for item in found)
    }


def events(directory):
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A process may have been killed while writing its final event.
    return result


def new_starts(path, position):
    """Read only new complete records while a worker appends to its journal."""
    starts = []
    if not path.exists():
        return starts, position
    with path.open("rb") as stream:
        stream.seek(position)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            position = stream.tell()
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if record["event"] == "start":
                starts.append(record)
    return starts, position


def outcomes(records):
    """Require a completed protocol; interrupted calls are classified by the parent."""
    by_node = defaultdict(list)
    result = {}
    for record in records:
        if record["event"] == "report":
            by_node[record["nodeid"]].append(record)
        if record["event"] != "finish":
            continue
        reports = by_node[record["nodeid"]]
        failures = [item for item in reports if item["outcome"] == "failed"]
        skips = [item for item in reports if item["outcome"] == "skipped"]
        calls = [item for item in reports if item["when"] == "call"]
        if failures:
            state = "error" if any(item["when"] != "call" for item in failures) else "failed"
        elif skips:
            state = "xfailed" if any(item.get("wasxfail") is not None for item in skips) else "skipped"
        elif calls:
            state = "xpassed" if calls[0].get("wasxfail") is not None else "passed"
        else:
            state = "error"
        result[record["nodeid"]] = state
    return result


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may outlive a parent that accepted SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_process(command, directory, cwd, environment, timeout):
    """Bound startup/collection and each active node, including its subprocesses."""
    save(
        directory / "command.json", {
            "argv": command, "cwd": str(cwd), "environment": {
                key: value
                for key, value in environment.items()
                if key.startswith(("TRITON_", "ASCEND_", "PYTEST_")) or key in ("PYTHONPATH", "TMPDIR",
                                                                                "TORCH_DEVICE_BACKEND_AUTOLOAD")
            }
        })
    started = time.monotonic()
    last_start = None
    deadline = started + timeout
    timed_out = False
    interrupted = False
    position = 0
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                starts, position = new_starts(directory / "events.jsonl", position)
                current = (starts[-1]["nodeid"], starts[-1]["time"]) if starts else last_start
                if current != last_start:
                    last_start = current
                    deadline = time.monotonic() + timeout
                if time.monotonic() >= deadline:
                    timed_out = True
                    stop_process(process)
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            interrupted = True
            stop_process(process)
        finally:
            stop_process(process)
    records = events(directory)
    finished = outcomes(records)
    starts = [item["nodeid"] for item in records if item["event"] == "start"]
    active = starts[-1] if starts and starts[-1] not in finished else None
    if active and not interrupted:
        if timed_out:
            finished[active] = "timeout"
        elif process.returncode < 0:
            finished[active] = "crash"
        elif process.returncode not in (0, 1):
            finished[active] = "error"
    result = {
        "exit_code": process.returncode, "timeout": timed_out, "interrupted": interrupted, "seconds":
        time.monotonic() - started, "last_started": starts[-1] if starts else None, "active_at_exit": active,
        "outcomes": finished
    }
    save(directory / "result.json", result)
    return result


def worker_environment(directory, device):
    environment = os.environ.copy()
    for name in ("cache", "dump", "tmp"):
        (directory / name).mkdir()
    environment.update({
        "ASCEND_RT_VISIBLE_DEVICES": str(device),
        "ASCEND_COMMUNITY_EVENTS": str(directory / "events.jsonl"),
        "ASCEND_COMMUNITY_NODES": str(directory / "nodes.json"),
        "PYTHONPATH": str(HERE) + os.pathsep + environment.get("PYTHONPATH", ""),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRITON_CACHE_DIR": str(directory / "cache"),
        "TRITON_DUMP_DIR": str(directory / "dump"),
        "TMPDIR": str(directory / "tmp"),
    })
    return environment


def pytest_command(directory, selectors, collect=False):
    command = [
        sys.executable, "-B", "-m", "pytest", "-p", "community_plugin", "--rootdir",
        str(REPOSITORY), "--device=npu", "-ra", "-v", "--tb=long", "-o", "addopts=", "-o",
        "cache_dir=" + str(directory / "pytest-cache"), "--junitxml=" + str(directory / "junit.xml")
    ]
    if collect:
        command.append("--collect-only")
    return command + selectors


def node_matches(node, selections):
    return not selections or any(node == item or node.startswith(item + "[") or node.startswith(item + "::")
                                 for item in selections)


def batches(nodes, definitions, size):
    grouped = defaultdict(list)
    for node in nodes:
        grouped[node.split("[", 1)[0]].append(node)
    for definition, values in grouped.items():
        step = 1 if definitions[definition]["isolate"] else size
        for start in range(0, len(values), step):
            yield values[start:start + step]


def progress(output, nodes, collection_errors, previous=None):
    completed = dict(previous["outcomes"]) if previous else {}
    attempts = list(previous["attempts"]) if previous else []
    recorded = {item["directory"] for item in attempts}
    for directory in sorted((output / "batches").glob("*")):
        if not directory.is_dir() or str(directory.relative_to(output)) in recorded:
            continue
        completed.update(outcomes(events(directory)))
        if (directory / "result.json").exists():
            result = read_json(directory / "result.json")
            completed.update(result["outcomes"])
            attempts.append({"directory": str(directory.relative_to(output)), **result})
    if set(completed) - set(nodes):
        raise ValueError("A batch reported nodes outside the saved collection")
    execution_errors = [
        attempt for attempt in attempts if attempt["timeout"] or attempt["interrupted"] or attempt["exit_code"] not in (
            0, 1) or (attempt["exit_code"] == 1 and not BAD_OUTCOMES.intersection(attempt["outcomes"].values()))
    ]
    report = {
        "nodes": len(nodes), "counts": dict(Counter(completed.values())), "outcomes": completed, "remaining":
        [node for node in nodes if node not in completed], "collection_errors": collection_errors, "execution_errors":
        execution_errors, "attempts": attempts
    }
    save(output / "progress.json", report)
    (output / "remaining.txt").write_text("".join(node + "\n" for node in report["remaining"]), encoding="utf-8")
    return report


def execute(args, manifest):
    output = args.output.resolve()
    if args.resume:
        if not (output / "identity.json").exists():
            raise ValueError("--resume requires a previously created output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "sources": source_identity(REPOSITORY), "select": args.select, "device": args.npu_device, "python":
            sys.executable, "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plugin": hashlib.sha256(
                (HERE / "community_plugin.py").read_bytes()).hexdigest()
        }
        if args.resume and read_json(output / "identity.json") != identity:
            raise ValueError("Source, runner, selection or device changed; use a new output directory")
        save(output / "identity.json", identity)
        definitions = {item["id"]: item for item in manifest["definitions"]}
        for selection in args.select:
            if not any(node_matches(item, [selection.split("[", 1)[0]]) for item in definitions):
                raise ValueError(f"Selection is not in the full community manifest: {selection}")
        (output / "collection").mkdir(exist_ok=True)
        (output / "batches").mkdir(exist_ok=True)
        selected_files = [
            path for path in manifest["files"]
            if not args.select or any(selection == path or selection.startswith(path + "::")
                                      for selection in args.select)
        ]
        nodes, collection_errors, collected_definitions = [], [], set()
        for path in selected_files:
            directory = output / "collection" / path.replace("/", "__")
            if args.resume and (directory / "result.json").exists():
                prior = read_json(directory / "result.json")
                if prior["timeout"] or prior["interrupted"] or (prior["exit_code"] or 0) < 0:
                    directory.rename(directory.with_name(directory.name + ".interrupted-" + str(time.time_ns())))
            if not (directory / "result.json").exists():
                # Interrupted collection gets a new attempt, without deleting the partial log.
                if directory.exists():
                    directory.rename(directory.with_name(directory.name + ".interrupted-" + str(time.time_ns())))
                directory.mkdir()
                result = run_process(pytest_command(directory, [path], collect=True), directory, REPOSITORY,
                                     worker_environment(directory, args.npu_device), args.node_timeout)
            else:
                result = read_json(directory / "result.json")
            if result["exit_code"] not in (0, 5) or result["timeout"] or result["interrupted"]:
                collection_errors.append({"file": path, **result})
            current = read_json(directory / "nodes.json") if (directory / "nodes.json").exists() else []
            for item in current:
                definition = item["nodeid"].split("[", 1)[0]
                if definition not in definitions:
                    raise ValueError(f"Collected an unlisted definition: {definition}")
                collected_definitions.add(definition)
                if node_matches(item["nodeid"], args.select):
                    nodes.append(item["nodeid"])
            if result["interrupted"] or result["timeout"] or (result["exit_code"] or 0) < 0:
                save(output / "collection_interruption.json", {"file": path, "result": result})
                break
        if len(nodes) != len(set(nodes)):
            raise ValueError("Duplicate parameter nodes collected")
        expected = {item for item in definitions if item.split("::", 1)[0] in selected_files}
        missing = sorted(expected - collected_definitions)
        if missing:
            collection_errors.append({"missing_definitions": missing})
        unmatched = [
            selection for selection in args.select if not any(node_matches(node, [selection]) for node in nodes)
        ]
        if unmatched:
            collection_errors.append({"unmatched_selections": unmatched})
        save(output / "nodes.json", nodes)
        report = progress(output, nodes, collection_errors)
        if args.collect_only or collection_errors:
            print(json.dumps({"collected_nodes": len(nodes), "collection_errors": collection_errors}, indent=2))
            return int(bool(collection_errors))
        for selected in batches(report["remaining"], definitions, args.batch_size):
            directory = output / "batches" / f"{time.time_ns()}-{hashlib.sha256(chr(10).join(selected).encode()).hexdigest()[:12]}"
            directory.mkdir()
            save(directory / "selected.json", selected)
            result = run_process(pytest_command(directory, selected), directory, REPOSITORY,
                                 worker_environment(directory, args.npu_device), args.node_timeout)
            report = progress(output, nodes, collection_errors, previous=report)
            if result["timeout"] or result["interrupted"] or result["exit_code"] not in (0, 1) or any(
                    node not in result["outcomes"] for node in selected):
                print("Execution stopped; inspect progress.json and device health before --resume.", file=sys.stderr)
                break
        print(json.dumps({"counts": report["counts"], "remaining": len(report["remaining"])}, indent=2))
        return int(
            bool(report["remaining"] or collection_errors or report["execution_errors"]
                 or BAD_OUTCOMES.intersection(report["counts"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Check the 458-definition manifest without importing NPU software")
    parser.add_argument("--output", type=Path, help="New directory for collection, JUnit, logs, cache and progress")
    parser.add_argument("--npu-device", type=int, help="Explicit physical NPU index; one supervisor uses one device")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue remaining nodes after checking the device; keep completed outcomes")
    parser.add_argument("--select", action="append", default=[],
                        help="Exact file, definition or parameter node; repeatable")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--node-timeout", type=float, default=180)
    args = parser.parse_args()
    manifest = read_json(HERE / "manifest.json")
    checked = check_scope(REPOSITORY, manifest)
    if args.check:
        print(json.dumps(checked, indent=2))
        return 0
    if args.output is None or args.npu_device is None:
        parser.error("--output and --npu-device are required")
    if args.npu_device < 0 or args.batch_size < 1 or args.node_timeout <= 0:
        parser.error("Device must be nonnegative; batch size and timeout must be positive")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    return execute(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
