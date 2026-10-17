"""CPU checks for accounting and interruption; no NPU modules are imported."""

import argparse
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("community_runner_under_test", TOOLS / "run.py")
plugin = load("community_plugin_under_test", TOOLS / "community_plugin.py")


def observe(tmp_path, source, timeout=10):
    file = tmp_path / "test_sample.py"
    file.write_text(source)
    output = tmp_path / "attempt"
    output.mkdir()
    environment = runner.worker_environment(output, 0)
    environment.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable, "-B", "-m", "pytest", "-p", "community_plugin", "--rootdir",
        str(tmp_path), "-o", "addopts=", "--junitxml=" + str(output / "junit.xml"),
        str(file)
    ]
    return runner.run_process(command, output, tmp_path, environment, timeout), output


def test_reported_outcomes_remain_distinct(tmp_path):
    result, output = observe(
        tmp_path, """
import pytest

def test_pass():
    assert 1 + 1 == 2

def test_failure():
    assert 2 == 3

@pytest.mark.skip(reason="original skip")
def test_skip():
    pass

@pytest.mark.xfail(reason="original xfail")
def test_xfail():
    assert False

@pytest.mark.xfail(reason="unexpected pass")
def test_xpass():
    pass

@pytest.fixture
def bad_setup():
    raise ValueError("setup failure")

def test_setup_error(bad_setup):
    pass

@pytest.fixture
def bad_teardown():
    yield
    raise ValueError("teardown failure")

def test_teardown_error(bad_teardown):
    pass
""")
    assert result["exit_code"] == 1
    assert Counter(result["outcomes"].values()) == {
        "passed": 1,
        "failed": 1,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
        "error": 2,
    }
    assert "assert 2 == 3" in (output / "stdout.log").read_text()
    assert (output / "junit.xml").exists()


def test_timeout_preserves_finished_and_unstarted_nodes(tmp_path):
    result, output = observe(
        tmp_path, """
import time
def test_completed():
    pass
def test_waiting():
    time.sleep(60)
def test_unstarted():
    raise AssertionError("must not run")
""", timeout=2)
    assert result["timeout"]
    assert result["outcomes"] == {"test_sample.py::test_completed": "passed", "test_sample.py::test_waiting": "timeout"}
    assert result["last_started"] == "test_sample.py::test_waiting"
    assert "test_sample.py::test_unstarted" not in runner.outcomes(runner.events(output))


def test_signal_exit_retains_failure_before_crash(tmp_path):
    result, _ = observe(
        tmp_path, """
import os
import signal
def test_failed():
    assert False
def test_terminated():
    os.kill(os.getpid(), signal.SIGTERM)
def test_unstarted():
    pass
""")
    assert result["exit_code"] == -15
    assert not result["timeout"]
    assert result["outcomes"] == {"test_sample.py::test_failed": "failed", "test_sample.py::test_terminated": "crash"}


@pytest.mark.parametrize("prior", [None, "0", "1"])
def test_npu_registration_preserves_child_autoload(monkeypatch, prior):
    if prior is None:
        monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    else:
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", prior)
    monkeypatch.delenv("ASCEND_COMMUNITY_EVENTS", raising=False)
    selected = []
    target = SimpleNamespace(backend="npu")
    modules = {
        "torch":
        SimpleNamespace(npu=SimpleNamespace(set_device=selected.append), __version__="test"),
        "triton":
        SimpleNamespace(
            runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target))),
            __version__="test", __file__="fake-triton"),
    }

    def import_module(name):
        if name == "torch_npu":
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            return SimpleNamespace()
        return modules[name]

    monkeypatch.setattr(plugin.importlib, "import_module", import_module)
    plugin.initialize_npu()
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == prior
    assert selected == [0]


@pytest.mark.parametrize("cuda,capability,expected,calls", [
    (False, None, True, 0),
    (True, 9, True, 1),
    (True, 10, False, 1),
])
def test_collection_guards_preserve_target_requirement(cuda, capability, expected, calls):
    expressions = []
    for name, functions in [
        ("test_matmul.py", {"test_blocked_scale_mxfp", "test_lhs_in_tmem", "test_lhs_in_tmem_mxfp"}),
        ("test_pipeliner.py", {"test_scatter_pipeline"}),
    ]:
        path = runner.REPOSITORY / "python/test/unit/language" / name
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                expressions.extend(
                    dec.args[0]
                    for dec in node.decorator_list
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "skipif")
    assert len(expressions) == 4
    for expression in expressions:
        queried = []

        def get_capability():
            queried.append(True)
            assert cuda, "A non-CUDA target queried CUDA capability"
            return (capability, 0)

        context = {
            "is_cuda": lambda: cuda, "torch":
            SimpleNamespace(cuda=SimpleNamespace(get_device_capability=get_capability))
        }
        assert eval(compile(ast.Expression(expression), "<guard>", "eval"), context) == expected
        assert len(queried) == calls


def test_full_manifest_and_historical_counts():
    import csv
    manifest = runner.read_json(TOOLS / "manifest.json")
    assert runner.check_scope(runner.REPOSITORY, manifest) == {"files": 49, "definitions": 458, "class_methods": 18}
    assert sum(item["device"] == "cpu" for item in manifest["definitions"]) == 18
    assert sum(item["isolate"] for item in manifest["definitions"]) == 31
    with (TOOLS / "a5_reference_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["definition"] for row in rows} == {item["id"] for item in manifest["definitions"]}
    assert sum(int(row["nodes"]) for row in rows) == 18922
    assert {key: sum(int(row[key])
                     for row in rows)
            for key in ("passed", "failed", "skipped", "xfailed", "timeout")} == {
                "passed": 7891,
                "failed": 5961,
                "skipped": 5059,
                "xfailed": 1,
                "timeout": 10,
            }
    assert all(not value
               for value in (sys.modules.get("torch"), sys.modules.get("torch_npu"), sys.modules.get("triton")))


def test_supervisor_collects_same_named_modules_and_resumes_without_retries(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    files = []
    definitions = []
    for directory, body in (("one", "assert False"), ("two", "assert True")):
        relative = f"python/test/unit/{directory}/test_same.py"
        path = repository / relative
        path.parent.mkdir(parents=True)
        path.write_text(f"def test_case():\n    {body}\n")
        files.append(relative)
        definitions.append({"id": relative + "::test_case", "device": "cpu", "isolate": False})
    (repository / "python/test/conftest.py").write_text("")
    manifest = {"files": files, "definitions": definitions}
    monkeypatch.setattr(runner, "REPOSITORY", repository)

    # Run real pytest workers with the real observer, but without the NPU option.
    original_command = runner.pytest_command

    def cpu_command(*args, **kwargs):
        return [value for value in original_command(*args, **kwargs) if value != "--device=npu"]

    monkeypatch.setattr(runner, "pytest_command", cpu_command)
    args = argparse.Namespace(output=tmp_path / "output", npu_device=0, resume=False, select=[], node_timeout=10,
                              batch_size=32, collect_only=True)
    assert runner.execute(args, manifest) == 0
    assert len(runner.read_json(args.output / "nodes.json")) == 2
    args.resume = True
    args.collect_only = False
    assert runner.execute(args, manifest) == 1
    report = runner.read_json(args.output / "progress.json")
    assert report["counts"] == {"failed": 1, "passed": 1}
    assert not report["remaining"]
    assert len(report["attempts"]) == 2
    assert runner.execute(args, manifest) == 1
    assert runner.read_json(args.output / "progress.json") == report


def test_risky_parameters_use_separate_process_batches():
    definitions = {"f.py::test_simple": {"isolate": False}, "f.py::test_lock": {"isolate": True}}
    nodes = ["f.py::test_simple[a]", "f.py::test_simple[b]", "f.py::test_lock[a]", "f.py::test_lock[b]"]
    assert list(runner.batches(nodes, definitions, 32)) == [nodes[:2], nodes[2:3], nodes[3:4]]


def test_partial_journal_line_is_read_after_completion(tmp_path):
    path = tmp_path / "events.jsonl"
    start = json.dumps({"event": "start", "nodeid": "f.py::test_item", "time": 1})
    path.write_text(start[:15])
    assert runner.new_starts(path, 0) == ([], 0)
    with path.open("a") as stream:
        stream.write(start[15:] + "\n")
    records, position = runner.new_starts(path, 0)
    assert records == [json.loads(start)]
    assert runner.new_starts(path, position) == ([], position)


def test_external_pytest_selection_does_not_hide_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x -k passing_only")
    monkeypatch.setenv("PYTEST_PLUGINS", "unrelated_environment_plugin")
    result, _ = observe(
        tmp_path, """
def test_failing_first():
    assert False
def test_failing_second():
    assert False
def test_passing_last():
    pass
""")
    assert Counter(result["outcomes"].values()) == {"failed": 2, "passed": 1}


def test_signal_after_last_test_is_not_a_successful_session(tmp_path):
    (tmp_path / "conftest.py").write_text("""
import os
import signal
def pytest_sessionfinish(session, exitstatus):
    os.kill(os.getpid(), signal.SIGTERM)
""")
    result, attempt = observe(tmp_path, "def test_completed():\n    pass\n")
    assert result["exit_code"] == -15
    assert result["outcomes"] == {"test_sample.py::test_completed": "passed"}
    ledger = tmp_path / "ledger"
    (ledger / "batches").mkdir(parents=True)
    attempt.rename(ledger / "batches/0")
    report = runner.progress(ledger, ["test_sample.py::test_completed"], [])
    assert not report["remaining"]
    assert report["counts"] == {"passed": 1}
    assert report["execution_errors"][0]["exit_code"] == -15


def test_interrupted_collection_can_resume_without_losing_its_log(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    path = repository / "python/test/unit/test_retry.py"
    path.parent.mkdir(parents=True)
    marker = repository / "first-import"
    path.write_text(f"""
from pathlib import Path
import time
marker = Path({str(marker)!r})
if not marker.exists():
    marker.touch()
    time.sleep(60)
def test_collected():
    pass
""")
    (repository / "python/test/conftest.py").write_text("")
    relative = str(path.relative_to(repository))
    manifest = {
        "files": [relative], "definitions": [{"id": relative + "::test_collected", "device": "cpu", "isolate": False}]
    }
    monkeypatch.setattr(runner, "REPOSITORY", repository)
    original_command = runner.pytest_command

    def cpu_command(*args, **kwargs):
        return [value for value in original_command(*args, **kwargs) if value != "--device=npu"]

    monkeypatch.setattr(runner, "pytest_command", cpu_command)
    args = argparse.Namespace(output=tmp_path / "output", npu_device=0, resume=False, select=[], node_timeout=2,
                              batch_size=32, collect_only=True)
    assert runner.execute(args, manifest) == 1
    args.resume = True
    assert runner.execute(args, manifest) == 0
    assert runner.read_json(args.output / "nodes.json") == [relative + "::test_collected"]
    archived = list((args.output / "collection").glob("*.interrupted-*"))
    assert len(archived) == 1
    assert runner.read_json(archived[0] / "result.json")["timeout"]
