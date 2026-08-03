import contextlib
import json
import os
import re
import threading
import warnings


ENV_VAR = "TRITON_KERNEL_ARGS_DUMP_DIR"

_counter = 0
_counter_lock = threading.Lock()
_tls = threading.local()


def _next_counter():
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _sanitize_filename(name):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return sanitized or "unnamed"


def _is_tensor_like(value):
    return hasattr(value, "cpu") and hasattr(value, "dtype") and hasattr(value, "data_ptr")


def _jsonable_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        converted = [_jsonable_value(item) for item in value]
        return converted if isinstance(value, list) else {"repr": repr(value), "items": converted}
    if isinstance(value, dict):
        return {
            str(k): _jsonable_value(v)
            for k, v in value.items()
            if isinstance(k, (bool, int, float, str))
        }
    return None


def _scalar_record(value):
    return {
        "type": type(value).__name__,
        "module": type(value).__module__,
        "value": _jsonable_value(value),
        "repr": repr(value),
    }


def _tensor_record(value, filename):
    record = {
        "type": type(value).__name__,
        "module": type(value).__module__,
        "filename": filename,
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
    }
    for attr in ("shape", "size"):
        if hasattr(value, attr):
            try:
                record[attr] = list(getattr(value, attr))
            except TypeError:
                pass
            except Exception as exc:
                record[f"{attr}_error"] = repr(exc)
    if hasattr(value, "stride"):
        try:
            record["stride"] = list(value.stride())
        except TypeError:
            try:
                record["stride"] = [value.stride(i) for i in range(len(record.get("shape", [])))]
            except Exception as exc:
                record["stride_error"] = repr(exc)
        except Exception as exc:
            record["stride_error"] = repr(exc)
    return record


def _warn(message):
    warnings.warn(f"kernel argument dump: {message}", RuntimeWarning, stacklevel=3)


def _dump_tensor(name, value, dump_dir):
    import torch

    filename = f"{_sanitize_filename(name)}.pt"
    path = os.path.join(dump_dir, filename)
    record = _tensor_record(value, filename)
    try:
        cpu_value = value.cpu()
    except Exception as exc:
        record["error"] = f"cpu failed: {exc!r}"
        _warn(f"failed to move argument {name!r} to CPU: {exc!r}")
        return record
    try:
        torch.save(cpu_value, path)
    except Exception as exc:
        record["error"] = f"torch.save failed: {exc!r}"
        _warn(f"failed to save argument {name!r} to {path!r}: {exc!r}")
    return record


def _dump_args_json(dump_dir, metadata):
    path = os.path.join(dump_dir, "args.json")
    try:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception as exc:
        _warn(f"failed to save args metadata to {path!r}: {exc!r}")


def is_enabled():
    return bool(os.getenv(ENV_VAR, "").strip())


def is_suppressed():
    return bool(getattr(_tls, "suppress", False))


@contextlib.contextmanager
def suppress_dump():
    previous = is_suppressed()
    _tls.suppress = True
    try:
        yield
    finally:
        _tls.suppress = previous


def dump_kernel_args(kernel_name, bound_args, warmup=False):
    root = os.getenv(ENV_VAR, "").strip()
    if not root or warmup or is_suppressed():
        return None

    counter = _next_counter()
    safe_kernel_name = _sanitize_filename(kernel_name)
    dump_dir = os.path.join(root, f"{safe_kernel_name}_pid_{os.getpid()}_call_{counter}")

    try:
        os.makedirs(dump_dir)
    except Exception as exc:
        _warn(f"failed to create dump directory {dump_dir!r}: {exc!r}")
        return None

    metadata = {
        "kernel_name": kernel_name,
        "pid": os.getpid(),
        "counter": counter,
        "arguments": {},
    }

    for name, value in bound_args.items():
        if _is_tensor_like(value):
            metadata["arguments"][name] = _dump_tensor(name, value, dump_dir)
        else:
            metadata["arguments"][name] = _scalar_record(value)

    _dump_args_json(dump_dir, metadata)
    return dump_dir
