import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass


class CompileTimeTracker:

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.timings = {}

    @contextmanager
    def record(self, name):
        start = self._clock()
        try:
            yield
        finally:
            self.timings[name] = self._clock() - start


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if v is not None
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=str)]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_safe(v)
            for k, v in sorted(vars(value).items())
            if not k.startswith("_") and v is not None
        }
    return str(value)


def _compact_json(value):
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _get_option_value(options, name):
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def collect_compile_config(src, options):
    return {
        "constants": _json_safe(getattr(src, "constants", {}) or {}),
        "multibuffer": _json_safe(_get_option_value(options, "multibuffer")),
    }


def resolve_kernel_name(src, metadata):
    if isinstance(metadata, Mapping):
        for key in ("kernel_name", "name"):
            value = metadata.get(key)
            if value:
                return value
    return getattr(src, "name", "<unknown>")


def build_ta_compile_time_log(kernel_name, config, timings, total_compile_time):
    if "stage_npubin" not in timings:
        return None
    npuir_compile_time = max(float(timings["stage_npubin"]), 0.0)
    ta_compile_time = max(float(total_compile_time) - npuir_compile_time, 0.0)
    return (
        f"[TA][{kernel_name}] "
        f"config={_compact_json(config)} "
        f"npuir_compile_time={npuir_compile_time:.6f}s "
        f"ta_compile_time={ta_compile_time:.6f}s"
    )


def emit_ta_compile_time_log(src, metadata, options, timings, total_compile_time, sink=print):
    line = build_ta_compile_time_log(
        resolve_kernel_name(src, metadata),
        collect_compile_config(src, options),
        timings,
        total_compile_time,
    )
    if line is not None:
        sink(line)
    return line
