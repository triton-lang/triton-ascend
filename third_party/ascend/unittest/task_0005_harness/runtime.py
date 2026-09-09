"""NPU launch helpers and artifact discovery for the task_0005 harness."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401 -- register torch.npu before runtime probes.
import triton

from .contracts import (
    COMPILE_MODE,
    LOGITS_PRIMARY,
    MergeCase,
    NormRopeCase,
    LogitsCase,
    TARGET_ARCH,
    ceil_div,
    expected_grid,
    fixture_path,
)

_TARGET_RE = re.compile(r'hacc\.target\s*=\s*#hacc\.target<"([^"]+)"')
_PARALLEL_RE = re.compile(r'parallel_mode\s*=\s*"([^"]+)"')
_MIX_RE = re.compile(r'mix_mode\s*=\s*"([^"]+)"')


@dataclass(frozen=True)
class LaunchResult:
    outputs: tuple[torch.Tensor, ...]
    grids: dict[str, tuple[int, ...]]
    kernel_names: tuple[str, ...]
    # A logical launch may emit more than one invocation of the same named
    # kernel (norm+RoPE launches Q then K).  Keep this explicit so a profiler
    # parser cannot silently discard half of the active samples.
    kernel_invocations: dict[str, int]


def _generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(seed)


@functools.lru_cache(maxsize=None)
def load_fixture(operator: str, variant: str = "before") -> Any:
    """Load a fixture by a package-relative file path with an isolated module name."""
    path = fixture_path(operator, variant)  # type: ignore[arg-type]
    module_name = f"task_0005_fixture_{operator}_{variant}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load fixture {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_merge_inputs(case: MergeCase, *, device: torch.device, seed: int = 20260905) -> dict[str, torch.Tensor]:
    generator = _generator(device, seed + case.num_splits)
    partial_out = torch.randn(
        case.partial_out_shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    partial_lse = torch.randn(
        case.partial_lse_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    # Exercise an individual -inf split while retaining a finite row so the
    # output stays a well-defined weighted merge.
    partial_lse[0, 0, 0] = -float("inf")
    return {"partial_out": partial_out, "partial_lse": partial_lse}


def make_norm_rope_inputs(case: NormRopeCase, *, device: torch.device, seed: int = 20260905) -> dict[str, torch.Tensor]:
    generator = _generator(device, seed + case.tokens)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16).contiguous()

    inverse_frequency = 1.0 / (10000.0**(torch.arange(0, case.rotary_dim, device=device, dtype=torch.float32) /
                                         case.rotary_dim))
    angles = torch.arange(case.table_tokens, device=device, dtype=torch.float32)[:, None] * inverse_frequency[None, :]
    return {
        "q":
        randn(case.q_shape),
        "k":
        randn(case.k_z_shape),
        "z":
        randn(case.k_z_shape),
        "q_weight":
        randn((case.head_dim, )),
        "k_weight":
        randn((case.head_dim, )),
        "k_bias":
        randn((case.head_dim, )),
        "cos":
        angles.cos().to(torch.bfloat16).contiguous(),
        "sin":
        angles.sin().to(torch.bfloat16).contiguous(),
        "positions":
        torch.randint(
            0,
            case.table_tokens,
            (case.tokens, ),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ).contiguous(),
    }


def make_logits_inputs(
    case: LogitsCase = LOGITS_PRIMARY,
    *,
    device: torch.device,
    seed: int = 20260905,
) -> dict[str, torch.Tensor]:
    generator = _generator(device, seed + case.seq_q + case.seq_k)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16).contiguous()

    return {
        "q": randn(case.q_shape),
        "weights": randn(case.weights_shape),
        "k": randn(case.k_shape),
    }


def _compile_kwargs(graph_optimize: bool, *, program_mapping_rule_mask: int = 0) -> dict[str, Any]:
    kwargs = {
        "compile_mode": COMPILE_MODE,
        "enable_graph_optimize": graph_optimize,
    }
    if program_mapping_rule_mask:
        kwargs["program_mapping_rule_mask"] = program_mapping_rule_mask
    return kwargs


def launch_merge_split(
    inputs: dict[str, torch.Tensor],
    case: MergeCase,
    *,
    graph_optimize: bool,
    program_mapping_rule_mask: int = 0,
) -> LaunchResult:
    module = load_fixture("merge_split")
    partial_out = inputs["partial_out"]
    partial_lse = inputs["partial_lse"]
    out = torch.empty(case.output_shape, device=partial_out.device, dtype=partial_out.dtype)
    lse = torch.empty((case.tokens, case.heads), device=partial_out.device, dtype=torch.float32)
    grid = expected_grid("merge_split", case)
    module._merge_split_states_kernel[grid](
        partial_out,
        partial_lse,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        case.num_splits,
        head_dim=case.head_dim,
        BLOCK_S=triton.next_power_of_2(case.num_splits),
        BLOCK_D=triton.next_power_of_2(case.head_dim),
        **_compile_kwargs(
            graph_optimize,
            program_mapping_rule_mask=program_mapping_rule_mask,
        ),
    )
    return LaunchResult(
        outputs=(out, lse),
        grids={"merge": grid},
        kernel_names=("_merge_split_states_kernel", ),
        kernel_invocations={"_merge_split_states_kernel": 1},
    )


def _launch_norm_specialization(
    module: Any,
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    case: NormRopeCase,
    num_heads: int,
    has_bias: bool,
    subtract_mean: bool,
    weight_bias: float,
    graph_optimize: bool,
    program_mapping_rule_mask: int,
) -> tuple[int, ...]:
    grid = (case.tokens, num_heads)
    module._indexer_norm_rope_kernel[grid](
        x,
        out,
        weight,
        bias,
        cos,
        sin,
        positions,
        x.stride(0),
        out.stride(0),
        cos.stride(0),
        1.0e-6,
        weight_bias,
        head_dim=case.head_dim,
        rotary_dim=case.rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=has_bias,
        BLOCK_D=triton.next_power_of_2(case.head_dim),
        BLOCK_R=triton.next_power_of_2(case.rotary_dim),
        **_compile_kwargs(
            graph_optimize,
            program_mapping_rule_mask=program_mapping_rule_mask,
        ),
    )
    return grid


def launch_norm_rope(
    inputs: dict[str, torch.Tensor],
    case: NormRopeCase,
    *,
    graph_optimize: bool,
    q_program_mapping_rule_mask: int = 0,
    k_program_mapping_rule_mask: int = 0,
) -> LaunchResult:
    module = load_fixture("norm_rope")
    q_out = torch.empty_like(inputs["q"])
    k_out = torch.empty_like(inputs["k"])
    q_grid = _launch_norm_specialization(
        module,
        inputs["q"],
        q_out,
        inputs["q_weight"],
        inputs["q_weight"],
        inputs["cos"],
        inputs["sin"],
        inputs["positions"],
        case=case,
        num_heads=case.q_heads,
        has_bias=False,
        subtract_mean=False,
        weight_bias=1.0,
        graph_optimize=graph_optimize,
        program_mapping_rule_mask=q_program_mapping_rule_mask,
    )
    k_grid = _launch_norm_specialization(
        module,
        inputs["k"],
        k_out,
        inputs["k_weight"],
        inputs["k_bias"],
        inputs["cos"],
        inputs["sin"],
        inputs["positions"],
        case=case,
        num_heads=case.k_heads,
        has_bias=True,
        subtract_mean=True,
        weight_bias=0.0,
        graph_optimize=graph_optimize,
        program_mapping_rule_mask=k_program_mapping_rule_mask,
    )
    return LaunchResult(
        outputs=(q_out, k_out, inputs["z"].contiguous()),
        grids={"q": q_grid, "k": k_grid},
        kernel_names=("_indexer_norm_rope_kernel", ),
        kernel_invocations={"_indexer_norm_rope_kernel": 2},
    )


def launch_indexer_logits(
    inputs: dict[str, torch.Tensor],
    case: LogitsCase = LOGITS_PRIMARY,
    *,
    graph_optimize: bool,
    program_mapping_rule_mask: int = 0,
) -> LaunchResult:
    module = load_fixture("indexer_logits")
    q = module.round_activations_e4m3(inputs["q"])
    k = module.round_activations_e4m3(inputs["k"])
    out = torch.empty(case.output_shape, device=q.device, dtype=torch.float32)
    grid = expected_grid("indexer_logits", case)
    module._indexer_logits_kernel[grid](
        q,
        k,
        inputs["weights"],
        out,
        case.seq_q,
        case.seq_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        inputs["weights"].stride(0),
        inputs["weights"].stride(1),
        out.stride(0),
        heads_per_group=case.heads_per_group,
        proxy_dim=case.proxy_dim,
        BLOCK_Q=case.block_q,
        BLOCK_K=case.block_k,
        **_compile_kwargs(
            graph_optimize,
            program_mapping_rule_mask=program_mapping_rule_mask,
        ),
    )
    return LaunchResult(
        outputs=(out, ),
        grids={"logits": grid},
        kernel_names=("_indexer_logits_kernel", ),
        kernel_invocations={"_indexer_logits_kernel": 1},
    )


def npu_available() -> bool:
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def npu_identity() -> dict[str, Any]:
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    identity: dict[str, Any] = {
        "physical_npu_env": physical,
        "runtime_available": npu_available(),
    }
    if npu_available():
        device_id = torch.npu.current_device()
        identity.update(
            runtime_device=device_id,
            runtime_name=torch.npu.get_device_name(device_id),
        )
    return identity


def compiler_identity() -> dict[str, Any]:
    """Resolve BiSheng through the installed backend in this exact process."""
    identity: dict[str, Any] = {
        "expected_path": "/home/w00609825/triton_workspace/.tasks/0904/0904_bisheng/bishengir/bin/bishengir-compile",
    }
    try:
        from triton.backends.ascend.utils import _get_npucompiler_path

        compiler = _get_npucompiler_path()
        path = Path(compiler[0] if isinstance(compiler, tuple) else compiler).resolve()
        identity["resolved_path"] = str(path)
        if path.is_file():
            identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity["matches_expected"] = (identity["resolved_path"] == identity["expected_path"]
                                        and identity.get("sha256")
                                        == "2e64447322192f909f6eb71a821527a41fe5f4e4b0064ed80059ff088bc8de74")
    except Exception as error:  # pragma: no cover - diagnostic boundary
        identity["resolution_error"] = f"{type(error).__name__}: {error}"
        identity["matches_expected"] = False
    return identity


def collect_cache_records(cache_root: Path, kernel_names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Discover exact cache entries; never infer an artifact from recency."""
    records: list[dict[str, Any]] = []
    if not cache_root.is_dir():
        return records
    for kernel_name in kernel_names:
        for manifest in sorted(cache_root.rglob(f"{kernel_name}.json")):
            try:
                data = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            child_paths = data.get("child_paths", {})
            files = {name: str(path) for name, path in child_paths.items() if isinstance(path, str)}
            # Newer Ascend cache manifests are flat: the compile options live
            # directly in ``<kernel>.json`` and sibling TTIR/TTAdapter files
            # share its directory.  Older manifests carry ``child_paths``.
            # Support both layouts so artifact collection is tied to the
            # exact manifest we found, never to a recency heuristic.
            if not files:
                files = {
                    path.name: str(path)
                    for path in sorted(manifest.parent.glob(f"{kernel_name}.*"))
                    if path.is_file()
                }
            adapter = next(
                (Path(path) for name, path in files.items() if name.endswith(".ttadapter")),
                None,
            )
            adapter_text = ""
            if adapter is not None and adapter.is_file():
                adapter_text = adapter.read_text(errors="replace")
            target = _TARGET_RE.search(adapter_text)
            parallel = _PARALLEL_RE.search(adapter_text)
            mix = _MIX_RE.search(adapter_text)
            manifest_target = data.get("target_arch")
            if not manifest_target and isinstance(data.get("target"), dict):
                manifest_target = data["target"].get("arch")
            manifest_parallel = data.get("parallel_mode")
            manifest_mix = data.get("mix_mode")
            manifest_compile_mode = data.get("compile_mode")
            records.append({
                "kernel_name":
                kernel_name,
                "manifest":
                str(manifest),
                "files":
                files,
                "target_arch":
                target.group(1) if target else manifest_target,
                "parallel_mode": (parallel.group(1) if parallel else manifest_parallel),
                "mix_mode":
                mix.group(1) if mix else manifest_mix,
                "compile_mode":
                manifest_compile_mode or COMPILE_MODE,
                "is_pure_simt":
                bool(data.get(
                    "is_pure_simt",
                    (parallel.group(1) if parallel else manifest_parallel) == "simt",
                )),
            })
    return records


def copy_cache_artifacts(records: list[dict[str, Any]], destination: Path) -> list[dict[str, Any]]:
    """Copy golden TTIR/cache metadata into an explicit run directory."""
    copied: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        output = dict(record)
        copied_files: dict[str, str] = {}
        for name, source_name in record["files"].items():
            source = Path(source_name)
            if not source.is_file():
                continue
            target = destination / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            copied_files[name] = target.name
        output["copied_files"] = copied_files
        copied.append(output)
    return copied


def validate_primary_metadata(records: list[dict[str, Any]]) -> list[str]:
    """Return discrepancies instead of hiding partial compilation evidence."""
    errors: list[str] = []
    if not records:
        return ["no exact kernel cache records found"]
    for record in records:
        if record.get("target_arch") != TARGET_ARCH:
            errors.append(f"{record['kernel_name']}: target={record.get('target_arch')!r}")
        if record.get("compile_mode") != COMPILE_MODE:
            errors.append(f"{record['kernel_name']}: compile_mode={record.get('compile_mode')!r}")
        if record.get("parallel_mode") not in ("simd", "mix_simd_simt"):
            errors.append(f"{record['kernel_name']}: parallel_mode={record.get('parallel_mode')!r}")
        if record.get("is_pure_simt"):
            errors.append(f"{record['kernel_name']}: unexpectedly pure SIMT")
    return errors
