import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
import torch

_HELPER_PATH = Path(__file__).resolve().parents[3] / "triton" / "runtime" / "kernel_args_dump.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("kernel_args_dump", _HELPER_PATH)
kernel_args_dump = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(kernel_args_dump)

ENV_VAR = kernel_args_dump.ENV_VAR
dump_kernel_args = kernel_args_dump.dump_kernel_args
suppress_dump = kernel_args_dump.suppress_dump


def _load_args_json(dump_dir):
    with open(os.path.join(dump_dir, "args.json")) as f:
        return json.load(f)


def test_kernel_args_dump_is_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)

    assert dump_kernel_args("kernel", {"x": torch.arange(4)}) is None
    assert list(tmp_path.iterdir()) == []


def test_kernel_args_dump_writes_tensors_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))

    x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    first = dump_kernel_args("my/kernel", {"x": x, "n": 4, "BLOCK": 8})
    second = dump_kernel_args("my/kernel", {"x": x, "n": 4, "BLOCK": 8})

    assert first is not None
    assert second is not None
    assert first != second
    assert re.match(r"my_kernel_pid_\d+_call_\d+$", os.path.basename(first))
    assert re.match(r"my_kernel_pid_\d+_call_\d+$", os.path.basename(second))
    assert int(os.path.basename(second).rsplit("_call_", 1)[1]) > int(os.path.basename(first).rsplit("_call_", 1)[1])

    dumped_x = torch.load(os.path.join(first, "x.pt"))
    assert torch.equal(dumped_x, x)

    metadata = _load_args_json(first)
    assert metadata["kernel_name"] == "my/kernel"
    assert metadata["arguments"]["x"]["filename"] == "x.pt"
    assert metadata["arguments"]["x"]["dtype"] == "torch.float32"
    assert metadata["arguments"]["x"]["shape"] == [2, 2]
    assert metadata["arguments"]["n"]["value"] == 4
    assert metadata["arguments"]["BLOCK"]["value"] == 8


def test_kernel_args_dump_skips_warmup_and_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))

    assert dump_kernel_args("kernel", {"x": torch.arange(1)}, warmup=True) is None
    with suppress_dump():
        assert dump_kernel_args("kernel", {"x": torch.arange(1)}) is None
    assert list(tmp_path.iterdir()) == []


def test_kernel_args_dump_warns_and_continues_on_tensor_save_failure(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))

    def fail_save(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.warns(RuntimeWarning, match="failed to save argument"):
        dump_dir = dump_kernel_args("kernel", {"x": torch.arange(2), "n": 2})

    metadata = _load_args_json(dump_dir)
    assert "torch.save failed" in metadata["arguments"]["x"]["error"]
    assert metadata["arguments"]["n"]["value"] == 2
