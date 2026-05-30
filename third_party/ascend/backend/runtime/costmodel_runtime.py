from __future__ import annotations

import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from triton.runtime.cache import get_cache_manager

try:
    from triton.runtime.cache import triton_key as _triton_key
except ImportError:
    _triton_key = None


def _costmodel_cache_namespace() -> str:
    """Return a stable hex key accepted by triton.runtime.cache manager."""
    if _triton_key is None:
        raw = "costmodel_release322"
    else:
        try:
            raw = f"costmodel_{_triton_key()}"
        except Exception:
            raw = "costmodel_release322"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_COSTMODEL_MEM_CACHE: Dict[str, float] = {}


def run_costmodel(ttir_or_path, extra_args=None, dump_ir_on_error=False):
    args = list(extra_args or [])
    if "-allow-unregistered-dialect" not in args:
        args.append("-allow-unregistered-dialect")

    from triton._C.libtriton import ascend as ascend_capi

    if os.path.exists(str(ttir_or_path)):
        with open(str(ttir_or_path), "r", encoding="utf-8") as f:
            mlir_text = f.read()
    else:
        mlir_text = ttir_or_path

    try:
        return ascend_capi.run_costmodel_inproc(mlir_text, args)
    except Exception as exc:
        if dump_ir_on_error and os.path.exists(str(ttir_or_path)):
            print(f"IR 文件: {ttir_or_path}")
        print(f"in-process costmodel failed: {exc}")
        return None


def get_costmodel_jobs(num_tasks: int) -> int:
    if num_tasks <= 1:
        return 1
    raw = os.environ.get("TRITON_COSTMODEL_WORKER_NUM")
    if raw is not None:
        try:
            parsed = int(raw)
            if parsed > 0:
                return min(parsed, num_tasks)
        except Exception:
            pass
    default_jobs = os.cpu_count() or 1
    return min(max(1, default_jobs), num_tasks)


def make_costmodel_cache_key(ttir: str, extra_args: Optional[List[str]]) -> str:
    h = hashlib.sha256()
    h.update(ttir.encode("utf-8"))
    h.update(b"|")
    if extra_args:
        h.update(" ".join(extra_args).encode("utf-8"))
    h.update(b"|")
    h.update(b"inproc_costmodel_v1")
    return h.hexdigest()


def load_costmodel_latency(cache_key: str) -> Optional[float]:
    cached = _COSTMODEL_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_manager = get_cache_manager(_costmodel_cache_namespace())
    file_name = f"{cache_key}.json"
    payload = cache_manager.get_file(file_name)
    if payload is None:
        return None

    try:
        parsed = json.loads(payload)
        latency = float(parsed["latency"])
        _COSTMODEL_MEM_CACHE[cache_key] = latency
        return latency
    except Exception:
        return None


def store_costmodel_latency(cache_key: str, latency: float) -> None:
    _COSTMODEL_MEM_CACHE[cache_key] = latency
    cache_manager = get_cache_manager(_costmodel_cache_namespace())
    file_name = f"{cache_key}.json"
    cache_manager.put(json.dumps({"latency": latency}), file_name, binary=False)


def parse_latency(stdout: str) -> float:
    import re

    match = re.search(r"Estimated Time:\s+([0-9.]+)\s*us", stdout)
    return float(match.group(1)) if match else float("inf")


def _warn_costmodel(msg: str) -> None:
    warnings.warn(f"costmodel_bench: {msg}", RuntimeWarning)


def _resolve_default_hardware_config() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../../costmodel/configs/ascend_910b.json"),
        os.path.join(os.path.dirname(__file__), "../../../../third_party/ascend/costmodel/configs/ascend_910b.json"),
        os.path.join(os.path.dirname(__file__), "../../../../../third_party/ascend/costmodel/configs/ascend_910b.json"),
    ]
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            return path
    return ""


def _build_costmodel_extra_args(arg_bindings: str, hardware_config: str = ""):
    base = "-ascend-perf-model"
    resolved_hardware_config = hardware_config or _resolve_default_hardware_config()
    # NOTE: Current inproc parser in installed runtime consumes only one payload
    # token after `-ascend-perf-model`. Frontend only forwards arg-bindings now.
    if arg_bindings:
        return [base, f"arg-bindings={arg_bindings}"]
    if resolved_hardware_config:
        return [base, f"hardware-config={resolved_hardware_config}"]
    return [base]


def _normalize_costmodel_items(config_ttir_items):
    pending_items = []
    costmodel_latencies = {}

    for item in config_ttir_items:
        if not isinstance(item, dict):
            continue
        config = item.get("config")
        ttir = item.get("ttir")
        arg_bindings = item.get("arg_bindings", "")
        hardware_config = item.get("hardware_config", "")
        if config is None:
            continue
        if not ttir:
            costmodel_latencies[config] = float("inf")
            continue
        pending_items.append((config, ttir, arg_bindings, hardware_config))

    return pending_items, costmodel_latencies


def _eval_one_costmodel_item(item):
    config, ttir, arg_bindings, hardware_config = item
    extra_args = _build_costmodel_extra_args(arg_bindings, hardware_config)
    cache_key = make_costmodel_cache_key(ttir, extra_args)
    cached = load_costmodel_latency(cache_key)
    if cached is not None:
        return config, cached

    output = run_costmodel(ttir_or_path=ttir, extra_args=extra_args)
    latency = float("inf") if output is None else parse_latency(output)
    store_costmodel_latency(cache_key, latency)
    return config, latency


def _evaluate_pending_items(pending_items, costmodel_latencies):
    if not pending_items:
        return

    jobs = get_costmodel_jobs(len(pending_items))
    if jobs <= 1:
        for item in pending_items:
            cfg, latency = _eval_one_costmodel_item(item)
            costmodel_latencies[cfg] = latency
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_eval_one_costmodel_item, item) for item in pending_items]
        for future in as_completed(futures):
            try:
                cfg, latency = future.result()
                costmodel_latencies[cfg] = latency
            except Exception:
                pass


def costmodel_bench(config_ttir_items):
    """Evaluate candidate configs with costmodel from prebuilt TTIR payloads.

    Args:
        config_ttir_items (Iterable[dict]): Iterable of per-config payloads.
            Each item should contain:
            - ``config``: config object used as result-map key.
            - ``ttir`` (str): TTIR text for costmodel evaluation.
            - ``arg_bindings`` (str, optional): Runtime bindings string passed
              to costmodel (for example ``"arg3=98432,pid_x=0"``).

    Returns:
        dict: Mapping ``{config: latency_us}``.
            Returns ``float("inf")`` for configs with missing/invalid TTIR or
            evaluation failures. Returns an empty dict for invalid/empty input.
    """
    try:
        items = list(config_ttir_items)
    except Exception:
        _warn_costmodel("config_ttir_items is not iterable; skip costmodel")
        return {}

    if len(items) == 0:
        return {}

    try:
        pending_items, costmodel_latencies = _normalize_costmodel_items(items)
        _evaluate_pending_items(pending_items, costmodel_latencies)

        for item in items:
            if isinstance(item, dict) and item.get("config") is not None:
                costmodel_latencies.setdefault(item["config"], float("inf"))

        return costmodel_latencies
    except Exception as exc:
        _warn_costmodel(f"unexpected failure: {exc}")
        fallback = {}
        for item in items:
            if isinstance(item, dict) and item.get("config") is not None:
                fallback[item["config"]] = float("inf")
        return fallback
