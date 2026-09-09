"""Unchanged-DSL grid-specialization observation contract for task_0005.

This module intentionally does *not* implement a graph rule.  It exercises the
real wrappers in ``example1/kernel_ori.py`` and records the boundary that
task_0002 v2 must later close: original grid evaluation, JIT specialization,
original-extent metadata, transform metadata, and the grid received by the
generated launcher.

The baseline represented here is explicit: the three new program-mapping bits
are off, so legacy JIT cache lookup precedes grid evaluation and no
grid-specialized extent attribute is allowed.  The same JSON schema also
contains the assertions a later enabled-bit implementation must satisfy.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from .contracts import LogitsCase, MergeCase, NormRopeCase, ceil_div
from .reference import (
    bf16_peak_ulp,
    max_abs_error,
    ref_indexer_logits,
    ref_indexer_norm_rope,
    ref_merge_split_states,
)
from .runtime import make_logits_inputs, make_merge_inputs, make_norm_rope_inputs

GRID_SPECIALIZATION_SCHEMA_VERSION = 1

# These values are owned by task_0001's append-only registry.  task_0005
# freezes them here so the harness can prove that the default legacy mask does
# not accidentally opt into a grid-specialized path before task_0002 v2.
GRID_SPECIALIZATION_RULE_BITS = {
    "IndependentAxisTensorizeRule": 512,
    "StaticProgramAxisFusionRule": 1024,
    "PersistentTaskStripMiningRule": 2048,
}
GRID_SPECIALIZATION_RULE_MASK = sum(GRID_SPECIALIZATION_RULE_BITS.values())
LEGACY_DEFAULT_GRAPH_RULE_MASK = 511

GRID_EVALUATION_AFTER_CACHE_LOOKUP = "after_cache_lookup"
GRID_EVALUATION_BEFORE_CACHE_LOOKUP = "before_cache_lookup"

# task_0002 v2 owns the final spelling of its compiler attr.  These markers
# make the baseline fail closed for all accepted spellings, while the on-state
# assertion requires a structured canonical value rather than trusting text.
GRID_EXTENT_ATTR_MARKERS = (
    "hacc.original_grid",
    "hacc.original_grid_extent",
    "hacc.grid_specialized_extent",
    "hacc.grid_specialization",
    "hacc.grid_specialization_extent",
)

ORIGINAL_WRAPPER_TARGETS = {
    "merge_split_states": "_merge_split_states_kernel",
    "indexer_norm_rope": "_indexer_norm_rope_kernel",
    "indexer_logits": "_indexer_logits_kernel",
}

GRID_OBSERVATION_MERGE_CASES = (
    MergeCase(num_splits=2, heads=64),
    MergeCase(num_splits=2, heads=65),
)
GRID_OBSERVATION_NORM_CASES = tuple(NormRopeCase(tokens=tokens) for tokens in (1, 16, 512))
GRID_OBSERVATION_LOGITS_CASES = tuple(LogitsCase(groups=groups) for groups in (1, 2, 4))


@dataclass(frozen=True)
class _CaseContext:
    label: str
    wrapper: str


def _task_root_from(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == "branches":
            return candidate.parent
    return None


def kernel_ori_path() -> Path:
    """Resolve the real task input rather than a copied fixture.

    Installed-wheel users must explicitly provide ``TASK0005_KERNEL_ORI_PATH``;
    source-worktree tests discover the sibling task input through the
    ``branches`` directory.  This keeps the dependency visible and avoids a
    hidden absolute import embedded in test code.
    """

    override = os.environ.get("TASK0005_KERNEL_ORI_PATH")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for origin in (Path(__file__).resolve(), Path.cwd().resolve()):
        root = _task_root_from(origin)
        if root is not None:
            candidates.append(root / "example1" / "kernel_ori.py")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates) or "no candidates"
    raise FileNotFoundError("could not resolve example1/kernel_ori.py; set TASK0005_KERNEL_ORI_PATH "
                            f"for an installed-wheel run (searched: {searched})")


def _module_name_for(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"task_0005_kernel_ori_{digest}"


def load_kernel_ori() -> Any:
    """Directly import the original DSL module under a stable isolated name."""

    path = kernel_ori_path()
    module_name = _module_name_for(path)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load original DSL module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _signature_names(jit_function: Any) -> tuple[str, ...]:
    function = getattr(jit_function, "fn", jit_function)
    return tuple(inspect.signature(function).parameters)


def original_dsl_contract() -> dict[str, Any]:
    """Return and validate source/wrapper provenance without compiling a kernel."""

    path = kernel_ori_path()
    module = load_kernel_ori()
    targets: dict[str, Any] = {}
    for wrapper, target_name in ORIGINAL_WRAPPER_TARGETS.items():
        target = getattr(module, target_name, None)
        wrapper_function = getattr(module, wrapper, None)
        if target is None or wrapper_function is None:
            raise AssertionError(f"original module is missing {wrapper} or {target_name}")
        signature = _signature_names(target)
        forbidden = {"num_heads", "tokens", "groups"} & set(signature)
        if forbidden:
            raise AssertionError(f"{target_name} changed unchanged-DSL signature with {sorted(forbidden)}")
        wrapper_source = inspect.getsource(wrapper_function)
        if target_name not in wrapper_source:
            # norm+RoPE intentionally keeps its Q/K policy in the public
            # wrapper and places the direct JIT launch in _norm_rope.  Follow
            # that existing wrapper edge instead of requiring a copied direct
            # launch solely for this harness.
            helper = getattr(module, "_norm_rope", None) if wrapper == "indexer_norm_rope" else None
            if helper is None or "_norm_rope" not in wrapper_source or target_name not in inspect.getsource(helper):
                raise AssertionError(f"{wrapper} no longer invokes {target_name}")
        targets[target_name] = {
            "wrapper": wrapper,
            "jit_signature": list(signature),
        }
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "targets": targets,
    }


def expected_grid_coverage() -> dict[str, list[list[int]]]:
    """The exact raw grids which task_0002 v2 must be able to distinguish."""

    return {
        "_merge_split_states_kernel": [[case.tokens, case.heads, 1] for case in GRID_OBSERVATION_MERGE_CASES],
        "_indexer_norm_rope_kernel":
        [[case.tokens, heads, 1] for case in GRID_OBSERVATION_NORM_CASES for heads in (case.q_heads, case.k_heads)],
        "_indexer_logits_kernel": [[
            ceil_div(case.seq_q, case.block_q),
            ceil_div(case.seq_k, case.block_k),
            case.groups,
        ] for case in GRID_OBSERVATION_LOGITS_CASES],
    }


def expected_enabled_assertion_contract() -> dict[str, Any]:
    """Stable on-state contract consumed by task_0002 v2's real injection tests."""

    return {
        "schema_version":
        GRID_SPECIALIZATION_SCHEMA_VERSION,
        "rule_bits":
        GRID_SPECIALIZATION_RULE_BITS,
        "required_event_fields": (
            "grid_evaluation",
            "specialization",
            "original_extent_attr",
            "transform_metadata",
            "actual_launch_grid",
        ),
        "required_relationships": {
            "grid_evaluation": ("grid_evaluation_before_cache_lookup when any of the three "
                                "program-mapping bits is enabled"),
            "specialization": ("canonical original grid is included in specialization/cache identity; "
                               "same grid reuses identity and different grid does not"),
            "original_extent_attr": ("present with canonical_grid equal to original_grid before GraphOptimize"),
            "transform_metadata": ("recorded for every launch; null is valid only for an eligible no-op"),
            "actual_launch_grid": ("captured at generated launcher and equals original grid when no "
                                   "transform is published"),
        },
        "coverage":
        expected_grid_coverage(),
    }


def grid_rule_mode(rule_mask: int = LEGACY_DEFAULT_GRAPH_RULE_MASK) -> dict[str, Any]:
    if isinstance(rule_mask, bool) or not isinstance(rule_mask, int) or rule_mask < 0:
        raise ValueError(f"invalid graph rule mask: {rule_mask!r}")
    enabled = {name: bool(rule_mask & bit) for name, bit in GRID_SPECIALIZATION_RULE_BITS.items()}
    return {
        "effective_rule_mask": rule_mask,
        "bits": enabled,
        "any_grid_specialization_bit_enabled": any(enabled.values()),
    }


def _canonical_grid(grid: Sequence[Any]) -> tuple[int, int, int]:
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Sequence):
        raise AssertionError(f"grid must be a 1-3 dimensional sequence, got {grid!r}")
    if not 1 <= len(grid) <= 3:
        raise AssertionError(f"grid must contain 1-3 dimensions, got {grid!r}")
    dimensions: list[int] = []
    for index, raw in enumerate(grid):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise AssertionError(f"invalid grid[{index}]={raw!r}")
        dimensions.append(raw)
    return tuple((dimensions + [1, 1, 1])[:3])  # type: ignore[return-value]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _identity_payload(value: Any) -> dict[str, str]:
    rendered = repr(value)
    return {
        "repr": rendered,
        "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }


def _legacy_grid_evaluation_phase() -> str:
    """Derive the baseline ordering from the active JIT implementation."""

    from triton.runtime.jit import JITFunction

    source = inspect.getsource(JITFunction.run)
    # Keep this tolerant of harmless local formatting and variable-name
    # changes between the source tree and an already-installed validation
    # wheel.  The call itself and the canonical callable-grid branch are the
    # ordering invariants we need to audit.
    cache_lookup = source.find("compute_cache_key(")
    grid_evaluation = source.find("if callable(grid)")
    if cache_lookup < 0 or grid_evaluation < 0:
        raise RuntimeError("could not locate cache/grid ordering in JITFunction.run")
    return (GRID_EVALUATION_AFTER_CACHE_LOOKUP
            if cache_lookup < grid_evaluation else GRID_EVALUATION_BEFORE_CACHE_LOOKUP)


def _metadata_payload(kernel: Any) -> dict[str, Any]:
    metadata = getattr(kernel, "metadata", None)
    if metadata is None:
        return {}
    asdict_method = getattr(metadata, "_asdict", None)
    raw = asdict_method() if callable(asdict_method) else vars(metadata)
    return _jsonable(raw)


def _extent_attr_observation(kernel: Any) -> dict[str, Any]:
    """Search exact compiled artifacts; never infer an attr from a grid value."""

    matches: dict[str, set[str]] = {}
    fragments: list[tuple[str, str]] = []
    for suffix, path in getattr(kernel, "metadata_group", {}).items():
        candidate = Path(path)
        if candidate.is_file():
            fragments.append((f"metadata_group:{suffix}", candidate.read_text(errors="replace")))
    for stage, payload in getattr(kernel, "asm", {}).items():
        text = payload.decode(errors="replace") if isinstance(payload, bytes) else str(payload)
        fragments.append((f"asm:{stage}", text))
    for source, text in fragments:
        for marker in GRID_EXTENT_ATTR_MARKERS:
            if marker in text:
                matches.setdefault(marker, set()).add(source)
    metadata = _metadata_payload(kernel)
    raw_specialization = metadata.get("program_grid_specialization")
    canonical_grid = None
    if isinstance(raw_specialization, Mapping):
        raw_grid = raw_specialization.get("grid")
        try:
            canonical_grid = list(_canonical_grid(raw_grid))
        except AssertionError:
            canonical_grid = None
    return {
        "present": bool(matches),
        "markers": sorted(matches),
        "sources": {marker: sorted(sources)
                    for marker, sources in sorted(matches.items())},
        # The compiled metadata is exported from the same C++-validated attr
        # before downstream lowering strips hacc.*.  Do not infer it from the
        # actual launch grid: only recorded compiler metadata is acceptable.
        "canonical_grid": canonical_grid,
    }


def _transform_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    transforms = metadata.get("program_grid_transforms")
    factor = metadata.get("coalesce_factor", 1)
    axis = metadata.get("coalesce_axis", -1)
    ceil_div = metadata.get("coalesce_grid_ceil_div", False)
    legacy_transform = isinstance(factor, int) and factor > 1 and axis in (0, 1, 2)
    return {
        "program_grid_transforms": _jsonable(transforms),
        "program_grid_transform_schema_version": metadata.get("program_grid_transform_schema_version"),
        "program_grid_transforms_cache_key": metadata.get("program_grid_transforms_cache_key"),
        "legacy_coalesce": {
            "factor": factor,
            "axis": axis,
            "ceil_div": ceil_div,
        },
        "present": transforms is not None or legacy_transform,
    }


class GridSpecializationObserver:
    """Observe selected original JITFunctions without changing their DSL bodies.

    The first launch of a compiled kernel cannot be intercepted retrospectively,
    because the launcher wrapper is created during that launch.  The harness
    deliberately launches every original wrapper twice: prime sequential grids,
    then repeat the same sequence.  The second pass captures the grid delivered
    to the generated launcher rather than merely assuming it from the wrapper.
    """

    def __init__(self, module: Any, *, rule_mask: int = LEGACY_DEFAULT_GRAPH_RULE_MASK):
        self.module = module
        self.mode = grid_rule_mode(rule_mask)
        self._grid_specialization_enabled = self.mode["any_grid_specialization_bit_enabled"]
        # Capture this before ``capture()`` monkey-patches JITFunction.run.
        self._legacy_grid_evaluation_phase = _legacy_grid_evaluation_phase()
        self._targets = {
            id(getattr(module, target_name)): target_name
            for target_name in ORIGINAL_WRAPPER_TARGETS.values()
        }
        self._context: _CaseContext | None = None
        self._active_events: list[dict[str, Any]] = []
        self._wrapped_launchers: dict[int, tuple[Any, Any]] = {}
        self.events: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def case(self, label: str, wrapper: str) -> Iterator[None]:
        previous = self._context
        self._context = _CaseContext(label=label, wrapper=wrapper)
        try:
            yield
        finally:
            self._context = previous

    def _event_before_run(
        self,
        jit_function: Any,
        target_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        grid: Any,
    ) -> tuple[dict[str, Any], Any]:
        from triton import knobs
        from triton.runtime import driver
        from triton.runtime.jit import compute_cache_key

        observed_kwargs = dict(kwargs)
        if self._grid_specialization_enabled:
            observed_kwargs.setdefault("program_mapping_rule_mask", self.mode["effective_rule_mask"])
        observed_kwargs["debug"] = (observed_kwargs.get("debug", getattr(jit_function, "debug", False))
                                    or knobs.runtime.debug)
        observed_kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        device = driver.active.get_current_device()
        kernel_cache, kernel_key_cache, _, backend, binder = jit_function.device_caches[device]
        bound_args, specialization, options = binder(*args, **observed_kwargs)
        prepared = None
        if self._grid_specialization_enabled:
            prepare = getattr(backend, "prepare_program_grid_specialization", None)
            if not callable(prepare):
                raise RuntimeError("enabled grid-specialization run has no backend pre-cache hook")
            prepared = prepare(grid, bound_args, options)
            if prepared is None:
                raise RuntimeError("enabled grid-specialization run did not resolve an original grid")
            original_grid, compiler_options = prepared
            if not isinstance(compiler_options, Mapping):
                raise RuntimeError("program-grid specialization hook returned invalid compiler options")
            options.update(compiler_options)
            observed_kwargs.update(compiler_options)
        cache_key = compute_cache_key(kernel_key_cache, specialization, options)
        phase = (GRID_EVALUATION_BEFORE_CACHE_LOOKUP
                 if self._grid_specialization_enabled else self._legacy_grid_evaluation_phase)
        context = self._context or _CaseContext(label="unlabeled", wrapper="unknown")
        event: dict[str, Any] = {
            "sequence": len(self.events) + 1,
            "case": context.label,
            "wrapper": context.wrapper,
            "jit_function": target_name,
            "original_grid": None,
            "grid_evaluation": {
                "phase":
                phase,
                "cache_lookup_order":
                ("cache_lookup_before_grid_evaluation"
                 if phase == GRID_EVALUATION_AFTER_CACHE_LOOKUP else "grid_evaluation_before_cache_lookup"),
                "source_kind":
                "callable" if callable(grid) else "tuple",
                "duration_ns":
                0,
            },
            "specialization": {
                "legacy_jit_cache_key": _identity_payload(cache_key),
                "binder_specialization": _identity_payload(specialization),
                "cache_hit_before_launch": cache_key in kernel_cache,
                "grid_in_specialization": self._grid_specialization_enabled,
            },
            "original_extent_attr": {
                "present": False,
                "markers": [],
                "sources": {},
                "canonical_grid": None,
            },
            "transform_metadata": {
                "program_grid_transforms": None,
                "program_grid_transform_schema_version": None,
                "program_grid_transforms_cache_key": None,
                "legacy_coalesce": {"factor": 1, "axis": -1, "ceil_div": False},
                "present": False,
            },
            "actual_launch_grid": None,
            "actual_launch_grid_observation": "not_intercepted",
        }

        if prepared is not None:
            event["original_grid"] = list(original_grid)
            return event, grid

        if callable(grid):
            original_callable = grid

            def observed_grid(bound: Any) -> Any:
                start = time.perf_counter_ns()
                evaluated = original_callable(bound)
                event["original_grid"] = list(_canonical_grid(evaluated))
                event["grid_evaluation"]["duration_ns"] = time.perf_counter_ns() - start
                return evaluated

            return event, observed_grid

        event["original_grid"] = list(_canonical_grid(grid))
        return event, grid

    def _wrap_launcher(self, kernel: Any) -> None:
        identity = id(kernel)
        if identity in self._wrapped_launchers:
            return
        original = getattr(kernel, "_run", None)
        if original is None:
            # The regular JIT path has already launched once before this point;
            # make the fallback explicit in case a backend changes that order.
            return

        def observed_launcher(*args: Any, **kwargs: Any) -> Any:
            if self._active_events and len(args) >= 3:
                event = self._active_events[-1]
                event["actual_launch_grid"] = list(_canonical_grid(args[:3]))
                event["actual_launch_grid_observation"] = "generated_launcher_intercept"
            return original(*args, **kwargs)

        kernel._run = observed_launcher
        self._wrapped_launchers[identity] = (kernel, original)

    @contextlib.contextmanager
    def capture(self) -> Iterator["GridSpecializationObserver"]:
        from triton.runtime.jit import JITFunction

        original_run = JITFunction.run

        def observed_run(jit_function: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
            target_name = self._targets.get(id(jit_function))
            if target_name is None or warmup:
                return original_run(jit_function, *args, grid=grid, warmup=warmup, **kwargs)
            forwarded_kwargs = dict(kwargs)
            if self._grid_specialization_enabled:
                forwarded_kwargs.setdefault("program_mapping_rule_mask", self.mode["effective_rule_mask"])
            event, forwarded_grid = self._event_before_run(jit_function, target_name, args, forwarded_kwargs, grid)
            self.events.append(event)
            self._active_events.append(event)
            try:
                kernel = original_run(jit_function, *args, grid=forwarded_grid, warmup=warmup, **forwarded_kwargs)
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                self._active_events.pop()
            if kernel is not None:
                metadata = _metadata_payload(kernel)
                event["compiled_kernel"] = {
                    "hash": getattr(kernel, "hash", None),
                    "name": getattr(kernel, "name", None),
                    "metadata": metadata,
                }
                event["original_extent_attr"] = _extent_attr_observation(kernel)
                event["transform_metadata"] = _transform_metadata(metadata)
                self._wrap_launcher(kernel)
            return kernel

        JITFunction.run = observed_run
        try:
            yield self
        finally:
            JITFunction.run = original_run
            for kernel, original_launcher in self._wrapped_launchers.values():
                if getattr(kernel, "_run", None) is not original_launcher:
                    kernel._run = original_launcher
            self._wrapped_launchers.clear()


def _case_label(operator: str, case: MergeCase | NormRopeCase | LogitsCase) -> str:
    if isinstance(case, MergeCase):
        return f"merge-h{case.heads}-s{case.num_splits}"
    if isinstance(case, NormRopeCase):
        return f"norm-t{case.tokens}-q{case.q_heads}-k{case.k_heads}"
    return f"logits-g{case.groups}-q{case.seq_q}-k{case.seq_k}"


def _call_original_wrapper(
    module: Any,
    operator: str,
    inputs: dict[str, torch.Tensor],
    case: MergeCase | NormRopeCase | LogitsCase,
) -> tuple[torch.Tensor, ...]:
    if operator == "merge_split":
        assert isinstance(case, MergeCase)
        return tuple(module.merge_split_states(inputs["partial_out"], inputs["partial_lse"]))
    if operator == "norm_rope":
        assert isinstance(case, NormRopeCase)
        return tuple(
            module.indexer_norm_rope(
                inputs["q"],
                inputs["k"],
                inputs["z"],
                inputs["q_weight"],
                inputs["k_weight"],
                inputs["k_bias"],
                inputs["cos"],
                inputs["sin"],
                inputs["positions"],
                head_dim=case.head_dim,
                num_q_heads=case.q_heads,
                num_k_heads=case.k_heads,
                rotary_dim=case.rotary_dim,
            ))
    assert operator == "indexer_logits"
    assert isinstance(case, LogitsCase)
    return (module.indexer_logits(
        inputs["q"],
        inputs["weights"],
        inputs["k"],
        block_q=case.block_q,
        block_k=case.block_k,
    ), )


def _correctness_payload(
    operator: str,
    inputs: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, ...],
) -> tuple[bool, dict[str, Any]]:
    if operator == "merge_split":
        expected_out, expected_lse = ref_merge_split_states(inputs["partial_out"], inputs["partial_lse"])
        out, lse = outputs
        result = {
            "output_max_abs_error": max_abs_error(out, expected_out),
            "lse_max_abs_error": max_abs_error(lse, expected_lse),
            "outputs_finite": bool(torch.isfinite(out).all() and torch.isfinite(lse).all()),
        }
        passed = bool(
            torch.allclose(out, expected_out, rtol=1.0e-2, atol=4.0e-3)
            and torch.allclose(lse, expected_lse, rtol=1.0e-3, atol=1.0e-3) and result["outputs_finite"])
        return passed, result
    if operator == "norm_rope":
        expected = ref_indexer_norm_rope(
            inputs["q"],
            inputs["k"],
            inputs["z"],
            inputs["q_weight"],
            inputs["k_weight"],
            inputs["k_bias"],
            inputs["cos"],
            inputs["sin"],
            inputs["positions"],
        )
        q_error = max_abs_error(outputs[0], expected[0])
        k_error = max_abs_error(outputs[1], expected[1])
        q_limit = 2.0 * bf16_peak_ulp(expected[0])
        k_limit = 2.0 * bf16_peak_ulp(expected[1])
        result = {
            "q_max_abs_error": q_error,
            "q_limit": q_limit,
            "k_max_abs_error": k_error,
            "k_limit": k_limit,
            "z_bitwise_passthrough": bool(torch.equal(outputs[2], expected[2])),
        }
        passed = bool(q_error <= q_limit and k_error <= k_limit and result["z_bitwise_passthrough"])
        return passed, result
    expected = ref_indexer_logits(inputs["q"], inputs["weights"], inputs["k"])
    actual = outputs[0]
    result = {
        "max_abs_error": max_abs_error(actual, expected),
        "outputs_finite": bool(torch.isfinite(actual).all()),
        "rtol": 1.0e-2,
        "atol": 5.0e-2,
    }
    passed = bool(
        torch.allclose(actual, expected, rtol=result["rtol"], atol=result["atol"]) and result["outputs_finite"])
    return passed, result


def _observation_cases(device: torch.device) -> list[tuple[str, Any, dict[str, torch.Tensor]]]:
    cases: list[tuple[str, Any, dict[str, torch.Tensor]]] = []
    cases.extend(("merge_split", case, make_merge_inputs(case, device=device)) for case in GRID_OBSERVATION_MERGE_CASES)
    cases.extend(
        ("norm_rope", case, make_norm_rope_inputs(case, device=device)) for case in GRID_OBSERVATION_NORM_CASES)
    cases.extend(
        ("indexer_logits", case, make_logits_inputs(case, device=device)) for case in GRID_OBSERVATION_LOGITS_CASES)
    return cases


def run_unchanged_dsl_grid_baseline(
    *,
    device: torch.device | None = None,
    rule_mask: int = LEGACY_DEFAULT_GRAPH_RULE_MASK,
) -> dict[str, Any]:
    """Run all raw-grid variants through the original public wrappers.

    This is intentionally a full-runtime helper.  It does not clear caches;
    callers must use a dedicated, clean ``TRITON_CACHE_DIR`` for an independent
    invocation, as documented by the task's kernel-run protocol.
    """

    if device is None:
        device = torch.device("npu")
    module = load_kernel_ori()
    observer = GridSpecializationObserver(module, rule_mask=rule_mask)
    case_entries = _observation_cases(device)
    outputs_by_case: dict[str, tuple[torch.Tensor, ...]] = {}

    # The repeated pass is intentional: it gives the observer a chance to wrap
    # the generated launcher after each first compile, then records the actual
    # grid it receives on the second direct-wrapper launch.
    with observer.capture():
        for pass_name in ("prime", "observe"):
            for operator, case, inputs in case_entries:
                wrapper = {
                    "merge_split": "merge_split_states",
                    "norm_rope": "indexer_norm_rope",
                    "indexer_logits": "indexer_logits",
                }[operator]
                label = _case_label(operator, case)
                with observer.case(f"{pass_name}:{label}", wrapper):
                    outputs = _call_original_wrapper(module, operator, inputs, case)
                if pass_name == "observe":
                    outputs_by_case[label] = outputs

    if device.type == "npu":
        torch.npu.synchronize()

    correctness: list[dict[str, Any]] = []
    for operator, case, inputs in case_entries:
        label = _case_label(operator, case)
        passed, details = _correctness_payload(operator, inputs, outputs_by_case[label])
        correctness.append({
            "operator": operator,
            "case": label,
            "parameters": asdict(case),
            "passed": passed,
            "checks": details,
        })

    report = {
        "schema_version": GRID_SPECIALIZATION_SCHEMA_VERSION,
        "mode": observer.mode,
        "source": original_dsl_contract(),
        "expected_enabled_assertion_contract": expected_enabled_assertion_contract(),
        "events": observer.events,
        "correctness": correctness,
        "passed": all(entry["passed"] for entry in correctness),
    }
    if observer.mode["any_grid_specialization_bit_enabled"]:
        assert_grid_specialization_enabled(report)
    else:
        assert_grid_specialization_off(report)
    return report


def _events_by_target(report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for event in report.get("events", []):
        target = event.get("jit_function")
        if isinstance(target, str):
            grouped.setdefault(target, []).append(event)
    return grouped


def _grid_tuple(event: Mapping[str, Any], field: str) -> tuple[int, int, int] | None:
    value = event.get(field)
    if value is None:
        return None
    try:
        return _canonical_grid(value)
    except AssertionError:
        return None


def assert_grid_specialization_off(report: Mapping[str, Any]) -> None:
    """Validate the current default-off baseline without guessing future behavior."""

    mode = report.get("mode", {})
    bits = mode.get("bits", {}) if isinstance(mode, Mapping) else {}
    if any(bits.get(name) for name in GRID_SPECIALIZATION_RULE_BITS):
        raise AssertionError(f"grid-specialization baseline unexpectedly enabled bits: {bits}")
    mask = mode.get("effective_rule_mask") if isinstance(mode, Mapping) else None
    if not isinstance(mask, int) or mask & GRID_SPECIALIZATION_RULE_MASK:
        raise AssertionError(f"baseline mask contains grid-specialization bits: {mask!r}")

    grouped = _events_by_target(report)
    expected = expected_grid_coverage()
    for target, expected_grids in expected.items():
        events = grouped.get(target, [])
        if not events:
            raise AssertionError(f"missing unchanged-DSL observations for {target}")
        observed = {_grid_tuple(event, "original_grid") for event in events}
        for grid in expected_grids:
            canonical = tuple(grid)
            if canonical not in observed:
                raise AssertionError(f"{target}: missing original grid {canonical}")
            matching = [event for event in events if _grid_tuple(event, "original_grid") == canonical]
            if not any(
                    _grid_tuple(event, "actual_launch_grid") == canonical
                    and event.get("actual_launch_grid_observation") == "generated_launcher_intercept"
                    for event in matching):
                raise AssertionError(f"{target}: did not intercept actual grid {canonical}")
        for event in events:
            evaluation = event.get("grid_evaluation", {})
            if evaluation.get("cache_lookup_order") != "cache_lookup_before_grid_evaluation":
                raise AssertionError(f"{target}: baseline changed JIT grid/cache order: {evaluation}")
            if event.get("original_extent_attr", {}).get("present"):
                raise AssertionError(f"{target}: baseline emitted extent attr: {event['original_extent_attr']}")
            transform = event.get("transform_metadata", {})
            if transform.get("present"):
                raise AssertionError(f"{target}: baseline emitted transform metadata: {transform}")
            actual = _grid_tuple(event, "actual_launch_grid")
            original = _grid_tuple(event, "original_grid")
            if actual is not None and actual != original:
                raise AssertionError(f"{target}: baseline transformed {original} to {actual}")

        # Legacy JIT identity may vary for constexprs (norm Q versus K), but
        # for every otherwise equal binder specialization, raw grid alone must
        # not split the cache key while all three new bits are off.
        by_specialization: dict[str, dict[str, set[tuple[int, int, int]]]] = {}
        for event in events:
            specialization = event.get("specialization", {})
            specialization_key = specialization.get("binder_specialization", {}).get("sha256")
            cache_key = specialization.get("legacy_jit_cache_key", {}).get("sha256")
            original = _grid_tuple(event, "original_grid")
            if not isinstance(specialization_key, str) or not isinstance(cache_key, str) or original is None:
                raise AssertionError(f"{target}: incomplete cache observation: {event}")
            bucket = by_specialization.setdefault(specialization_key, {})
            bucket.setdefault(cache_key, set()).add(original)
        for specialization_key, cache_entries in by_specialization.items():
            all_grids = set().union(*cache_entries.values())
            if len(all_grids) > 1 and len(cache_entries) != 1:
                raise AssertionError(f"{target}: grid split legacy cache under specialization "
                                     f"{specialization_key}: {cache_entries}")

    if not all(entry.get("passed") for entry in report.get("correctness", [])):
        raise AssertionError("unchanged-DSL baseline correctness failed")


def assert_grid_specialization_enabled(report: Mapping[str, Any]) -> None:
    """Assertion interface for task_0002 v2 after it turns on a new bit.

    It intentionally does not require a transform on every kernel: a valid
    no-op still has to carry the original extent and use a grid-specialized
    cache identity.  The later task supplies ``canonical_grid`` after parsing
    its chosen versioned compiler attr.
    """

    mode = report.get("mode", {})
    if not isinstance(mode, Mapping) or not mode.get("any_grid_specialization_bit_enabled"):
        raise AssertionError("enabled assertion requires at least one program-mapping bit")
    grouped = _events_by_target(report)
    for target, expected_grids in expected_grid_coverage().items():
        events = grouped.get(target, [])
        for grid in expected_grids:
            matching = [event for event in events if _grid_tuple(event, "original_grid") == tuple(grid)]
            if not matching:
                raise AssertionError(f"{target}: missing enabled observation for {grid}")
            for event in matching:
                evaluation = event.get("grid_evaluation", {})
                if evaluation.get("cache_lookup_order") != "grid_evaluation_before_cache_lookup":
                    raise AssertionError(f"{target}: grid was not evaluated before cache lookup")
                specialization = event.get("specialization", {})
                if not specialization.get("grid_in_specialization"):
                    raise AssertionError(f"{target}: cache identity omitted original grid")
                extent = event.get("original_extent_attr", {})
                if not extent.get("present") or _grid_tuple(extent, "canonical_grid") != tuple(grid):
                    raise AssertionError(f"{target}: original extent attr does not match grid {grid}: {extent}")
            # The prime pass creates the launcher and cannot be intercepted
            # retrospectively.  The subsequent observe pass must intercept at
            # least one launch for this exact original grid.
            if not any(
                    _grid_tuple(event, "actual_launch_grid") == tuple(grid)
                    and event.get("actual_launch_grid_observation") == "generated_launcher_intercept"
                    for event in matching):
                raise AssertionError(f"{target}: actual launcher grid was not captured for {grid}")

        key_by_grid: dict[tuple[int, int, int], set[str]] = {}
        for event in events:
            original = _grid_tuple(event, "original_grid")
            cache_key = event.get("specialization", {}).get("legacy_jit_cache_key", {}).get("sha256")
            if original is not None and isinstance(cache_key, str):
                key_by_grid.setdefault(original, set()).add(cache_key)
        for grid, keys in key_by_grid.items():
            if len(keys) != 1:
                raise AssertionError(f"{target}: same grid {grid} did not reuse one identity: {keys}")
        for first_grid, first_keys in key_by_grid.items():
            for second_grid, second_keys in key_by_grid.items():
                if first_grid != second_grid and first_keys == second_keys:
                    raise AssertionError(f"{target}: distinct grids share enabled cache identity: "
                                         f"{first_grid} and {second_grid}")
