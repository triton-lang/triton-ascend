from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import warnings
from collections.abc import Mapping
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

_ASCEND_910B_CONFIG = "ascend_910b.json"
_DAVID_V100_CONFIG = "ascend_davidv100.json"


def _normalize_soc_version(target_arch) -> str:
    return re.sub(r"[^a-z0-9]", "", str(target_arch or "").lower())


def _costmodel_config_filename_for_arch(target_arch) -> str:
    """Map runtime/compile target SoC names to the available model profiles."""
    arch = _normalize_soc_version(target_arch)
    if not arch:
        # Preserve the low-level costmodel_bench default for callers that do
        # not have an active driver. Autotune always supplies backend.target.
        return _ASCEND_910B_CONFIG

    # A5/David SoCs. rtGetSocVersion may report product names such as
    # Ascend910_9589 or Ascend950PR_9579 for the same model family.
    if arch.startswith(("ascend91095", "ascend950", "davidv100", "david100", "dv100")):
        return _DAVID_V100_CONFIG

    # A2/A3 SoCs. Product strings include Ascend910B[1-4] and internal
    # rtGetSocVersion names such as Ascend910_9362.
    if arch.startswith(("ascend910b", "ascend91093")):
        return _ASCEND_910B_CONFIG

    raise ValueError(f"Costmodel has no hardware profile for target arch {target_arch!r}")


def _costmodel_config_candidates(filename: str):
    runtime_dir = os.path.dirname(__file__)
    return [
        # Stable installed-wheel path:
        # triton/backends/ascend/costmodel/configs/<profile>.json
        os.path.join(runtime_dir, "../costmodel/configs", filename),
        # Source-tree path used by editable/development installs.
        os.path.join(runtime_dir, "../../costmodel/configs", filename),
        # Compatibility fallbacks for existing source/build layouts.
        os.path.join(runtime_dir, "../../../costmodel/configs", filename),
        os.path.join(runtime_dir, "../../../../third_party/ascend/costmodel/configs", filename),
        os.path.join(runtime_dir, "../../../../../third_party/ascend/costmodel/configs", filename),
    ]


def _validate_hardware_config(path: str) -> str:
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Costmodel hardware config does not exist: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Costmodel hardware config {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Costmodel hardware config {resolved}: top level must be an object")
    return resolved


def _resolve_hardware_config(hardware_config: str = "", target_arch="") -> str:
    if hardware_config:
        return _validate_hardware_config(hardware_config)

    filename = _costmodel_config_filename_for_arch(target_arch)
    for candidate in _costmodel_config_candidates(filename):
        path = os.path.realpath(os.path.abspath(candidate))
        if os.path.isfile(path):
            return _validate_hardware_config(path)
    raise FileNotFoundError(f"Costmodel hardware profile {filename!r} was not found for target arch {target_arch!r}")


def _get_current_target_arch() -> str:
    try:
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        return str(getattr(target, "arch", "") or "")
    except Exception:
        return ""


def _canonical_binding_key(key) -> str:
    key = str(key).strip()
    if not key:
        raise ValueError("Costmodel binding key must not be empty")
    lower = key.lower()
    if lower.isdigit():
        return f"arg{int(lower)}"
    aliases = {
        "pidx": "pid_x",
        "pidy": "pid_y",
        "pidz": "pid_z",
        "program_id_x": "pid_x",
        "program_id_y": "pid_y",
        "program_id_z": "pid_z",
        "num_programsx": "num_programs_x",
        "num_programsy": "num_programs_y",
        "num_programsz": "num_programs_z",
    }
    return aliases.get(lower, lower)


def _format_binding_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    value = str(value).strip()
    if not value:
        raise ValueError("Costmodel binding value must not be empty")
    if re.fullmatch(r"[+-]?\d+", value):
        return str(int(value))
    return value


def _canonicalize_arg_bindings(arg_bindings) -> str:
    if not arg_bindings:
        return ""
    if isinstance(arg_bindings, Mapping):
        pairs = arg_bindings.items()
    elif isinstance(arg_bindings, str):
        parsed = []
        for pair in arg_bindings.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid Costmodel binding {pair!r}; expected key=value")
            parsed.append(pair.split("=", 1))
        pairs = parsed
    else:
        raise TypeError("Costmodel arg_bindings must be a mapping or key=value string")

    normalized = {}
    for key, value in pairs:
        normalized[_canonical_binding_key(key)] = _format_binding_value(value)
    return ",".join(f"{key}={normalized[key]}" for key in sorted(normalized))


def _hardware_config_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def make_costmodel_cache_key(
    ttir: str,
    extra_args: Optional[List[str]] = None,
    *,
    arg_bindings="",
    hardware_config: str = "",
    target_arch: str = "",
) -> str:
    h = hashlib.sha256()
    h.update(ttir.encode("utf-8"))
    h.update(b"|")
    if arg_bindings or hardware_config or target_arch:
        h.update(_canonicalize_arg_bindings(arg_bindings).encode("utf-8"))
        h.update(b"|")
        h.update(str(target_arch).encode("utf-8"))
        h.update(b"|")
        if hardware_config:
            h.update(_hardware_config_digest(hardware_config).encode("ascii"))
    elif extra_args:
        # Backwards-compatible keying for external callers of this helper.
        h.update(" ".join(extra_args).encode("utf-8"))
    h.update(b"|")
    h.update(b"inproc_costmodel_v3_soc_profiled")
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


def _resolve_default_hardware_config(target_arch="") -> str:
    return _resolve_hardware_config(target_arch=target_arch)


def _build_costmodel_extra_args(arg_bindings, hardware_config: str = "", target_arch=""):
    base = "-ascend-perf-model"
    canonical_bindings = _canonicalize_arg_bindings(arg_bindings)
    resolved_hardware_config = _resolve_hardware_config(hardware_config, target_arch)
    result = [base]
    if canonical_bindings:
        result.append(f"arg-bindings={canonical_bindings}")
    result.append(f"hardware-config={resolved_hardware_config}")
    return result


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
        target_arch = item.get("target_arch", "") or _get_current_target_arch()
        if config is None:
            continue
        if not ttir:
            costmodel_latencies[config] = float("inf")
            continue
        try:
            canonical_bindings = _canonicalize_arg_bindings(arg_bindings)
            resolved_hardware_config = _resolve_hardware_config(hardware_config, target_arch)
        except Exception as exc:
            _warn_costmodel(f"failed to resolve inputs for config {config}: {type(exc).__name__}: {exc}")
            costmodel_latencies[config] = float("inf")
            continue
        pending_items.append((config, ttir, canonical_bindings, resolved_hardware_config, target_arch))

    return pending_items, costmodel_latencies


def _eval_one_costmodel_item(item):
    config, ttir, arg_bindings, hardware_config, target_arch = item
    extra_args = _build_costmodel_extra_args(arg_bindings, hardware_config, target_arch)
    cache_key = make_costmodel_cache_key(
        ttir,
        arg_bindings=arg_bindings,
        hardware_config=hardware_config,
        target_arch=target_arch,
    )
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


def _unwrap_jit_function(fn):
    from triton.runtime.jit import JITFunction

    current = fn
    visited = set()
    while not isinstance(current, JITFunction):
        obj_id = id(current)
        if obj_id in visited or not hasattr(current, "fn"):
            raise TypeError("Costmodel pruning requires a triton.runtime.JITFunction")
        visited.add(obj_id)
        current = current.fn
    return current


def _materialize_ttir_for_config(fn, config, named_args, runtime_kwargs):
    """Materialize optimized TTIR through the same binder used by JIT run()."""
    from triton._C.libtriton import ir
    from triton.runtime.driver import driver

    jit_fn = _unwrap_jit_function(fn)
    positional_args = tuple((named_args or {}).values())
    candidate_kwargs = dict(runtime_kwargs or {})
    grid = candidate_kwargs.pop("grid", None)
    candidate_kwargs.pop("warmup", None)

    conflicts = candidate_kwargs.keys() & config.kwargs.keys()
    if conflicts:
        raise ValueError(f"Conflicting meta-parameters: {', '.join(sorted(conflicts))}")
    candidate_kwargs.update(config.all_kwargs())

    device = driver.active.get_current_device()
    _, _, target, active_backend, binder = jit_fn.device_caches[device]
    bound_args, specialization, raw_options = binder(*positional_args, **candidate_kwargs)
    options, signature, constexprs, attrs = jit_fn._pack_args(active_backend, candidate_kwargs, bound_args,
                                                              specialization, raw_options)

    src = jit_fn.ASTSource(jit_fn, signature, constexprs, attrs)
    context = ir.context()
    ir.load_dialects(context)
    active_backend.load_dialects(context)
    module = src.make_ir(
        target,
        options,
        active_backend.get_codegen_implementation(options),
        active_backend.get_module_map(),
        context,
    )

    stages = {}
    active_backend.add_stages(stages, options, src.language)
    ttir_stage = stages.get("ttir")
    if ttir_stage is not None:
        module = ttir_stage(module, {})

    if callable(grid):
        grid = grid(bound_args)
    return str(module), bound_args, signature, grid


def _try_parse_int_binding(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if hasattr(value, "value") and isinstance(value.value, numbers.Integral):
        return int(value.value)
    if hasattr(value, "dtype") or hasattr(value, "data_ptr"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _build_costmodel_arg_bindings(ttir_text, bound_args, signature, grid):
    """Bind visible TTIR scalar args and program dimensions for one config."""
    match = re.search(r"tt\.func\s+(?:public\s+)?@\w+\((.*?)\)\s*(?:attributes|\{)", ttir_text, re.S)
    if match is None:
        raise ValueError("Unable to locate the TTIR entry function signature")

    # Costmodel bindings use positional argN keys, while TTIR SSA arguments
    # may either use generic names (%arg0) or preserve source names (%n_rows).
    # Enumerate every entry argument instead of depending on its SSA spelling.
    ttir_arg_count = len(re.findall(r"%[-a-zA-Z$._0-9]+\s*:", match.group(1)))
    # The JIT binder marks both tl.constexpr parameters and specialized
    # runtime constants (for example an integer value of 1 or None) as
    # constexpr in signature. Skip them so source parameters stay aligned
    # with the runtime-only TTIR argument list.
    visible_names = [name for name, ty in signature.items() if ty != "constexpr"]
    bindings = {}
    for arg_id, name in zip(range(ttir_arg_count), visible_names):
        value = _try_parse_int_binding(bound_args.get(name))
        if value is not None:
            bindings[f"arg{arg_id}"] = value

    grid_values = ()
    if grid is not None:
        if not isinstance(grid, (tuple, list)):
            grid = (grid, )
        grid_values = tuple(_try_parse_int_binding(x) for x in grid)

    for index, dim in enumerate(("x", "y", "z")):
        if f"tt.get_program_id {dim}" in ttir_text:
            bindings[f"pid_{dim}"] = 0
        if f"tt.get_num_programs {dim}" in ttir_text:
            if index >= len(grid_values) or grid_values[index] is None:
                raise ValueError(f"Unable to resolve num_programs_{dim} from grid")
            bindings[f"num_programs_{dim}"] = grid_values[index]

    return _canonicalize_arg_bindings(bindings)


def _build_costmodel_items(fn, configs, named_args, runtime_kwargs, hardware_config="", target_arch=""):
    items = []
    equivalence_keys = {}
    for config in configs:
        try:
            ttir, bound_args, signature, grid = _materialize_ttir_for_config(fn, config, named_args, runtime_kwargs)
            arg_bindings = _build_costmodel_arg_bindings(ttir, bound_args, signature, grid)
            item = {
                "config": config,
                "ttir": ttir,
                "arg_bindings": arg_bindings,
                "hardware_config": hardware_config,
                "target_arch": target_arch,
            }
            equivalence_keys[config] = make_costmodel_cache_key(
                ttir,
                arg_bindings=arg_bindings,
                hardware_config=hardware_config,
                target_arch=target_arch,
            )
        except Exception as exc:
            _warn_costmodel(f"failed to prepare config {config}: {type(exc).__name__}: {exc}")
            item = {
                "config": config,
                "ttir": "",
                "arg_bindings": "",
                "hardware_config": hardware_config,
                "target_arch": target_arch,
            }
        items.append(item)
    return items, equivalence_keys


def _resolve_keep_count(config_count, top_k):
    if isinstance(top_k, bool):
        raise TypeError("costmodel top_k must be an int or a float in (0, 1]")
    if isinstance(top_k, float):
        if not 0.0 < top_k <= 1.0:
            raise ValueError("costmodel top_k ratio must be in (0, 1]")
        keep_count = math.ceil(config_count * top_k)
    elif isinstance(top_k, int):
        if top_k <= 0:
            raise ValueError("costmodel top_k must be positive")
        keep_count = top_k
    else:
        raise TypeError("costmodel top_k must be an int or a float in (0, 1]")
    return min(config_count, keep_count)


def _select_costmodel_configs(configs, latency_map, equivalence_keys, top_k=0.25):
    """Select primary/fallback configs while preserving TTIR-equivalent groups."""
    configs = list(configs)
    original_index = {config: index for index, config in enumerate(configs)}
    valid = [config for config in configs if math.isfinite(float(latency_map.get(config, float("inf"))))]
    if not valid:
        return configs, []

    keep_count = _resolve_keep_count(len(configs), top_k)
    groups = {}
    for config in valid:
        key = equivalence_keys.get(config, ("config", original_index[config]))
        groups.setdefault(key, []).append(config)

    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            min(float(latency_map[config]) for config in group),
            min(original_index[config] for config in group),
        ),
    )

    primary = []
    for group in ranked_groups:
        if len(primary) >= keep_count:
            break
        primary.extend(group)

    if len(primary) < keep_count:
        primary_set = set(primary)
        primary.extend(config for config in configs if config not in primary_set)
        primary = primary[:keep_count]

    primary_set = set(primary)
    ranked_valid = [config for group in ranked_groups for config in group]
    fallback = [config for config in ranked_valid if config not in primary_set]
    fallback.extend(config for config in configs if config not in primary_set and config not in fallback)
    return primary, fallback


def prune_configs_by_costmodel(backend, fn, configs, named_args, runtime_kwargs, options=None):
    """Ascend backend hook used by Triton's generic autotuner."""
    options = dict(options or {})
    configs = list(configs)
    if len(configs) <= 1:
        return configs, []

    top_k = options.get("top_k", 0.25)
    # Validate user options before resolving profiles or generating TTIR.
    _resolve_keep_count(len(configs), top_k)

    target_arch = str(getattr(getattr(backend, "target", None), "arch", "") or "")
    hardware_config = _resolve_hardware_config(options.get("hardware_config", "") or "", target_arch)
    items, equivalence_keys = _build_costmodel_items(
        fn,
        configs,
        named_args,
        runtime_kwargs,
        hardware_config=hardware_config,
        target_arch=target_arch,
    )
    latency_map = costmodel_bench(items)
    if not isinstance(latency_map, dict):
        return configs, []

    return _select_costmodel_configs(
        configs,
        latency_map,
        equivalence_keys,
        top_k=top_k,
    )


def costmodel_bench(config_ttir_items):
    """Evaluate candidate configs with costmodel from prebuilt TTIR payloads.

    Args:
        config_ttir_items (Iterable[dict]): Iterable of per-config payloads.
            Each item should contain:
            - ``config``: config object used as result-map key.
            - ``ttir`` (str): TTIR text for costmodel evaluation.
            - ``arg_bindings`` (str, optional): Runtime bindings string passed
              to costmodel (for example ``"arg3=98432,pid_x=0"``).
            - ``hardware_config`` (str, optional): Explicit hardware profile
              path. It overrides target-based profile selection.
            - ``target_arch`` (str, optional): SoC name used to select a
              packaged profile. Defaults to the active Triton target.

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
