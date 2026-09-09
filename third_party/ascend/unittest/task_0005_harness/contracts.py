"""Frozen contracts shared by the task_0005 unit, lit, and NPU test layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

TARGET_ARCH = "Ascend950PR_9579"
COMPILE_MODE = "simd_simt_template"
PHYSICAL_NPU = 7

PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = PACKAGE_ROOT / "fixtures"

OperatorName = Literal["merge_split", "norm_rope", "indexer_logits"]
VariantName = Literal["before", "after"]


@dataclass(frozen=True)
class MergeCase:
    num_splits: int
    tokens: int = 8
    heads: int = 64
    head_dim: int = 192

    @property
    def partial_out_shape(self) -> tuple[int, int, int, int]:
        return self.num_splits, self.tokens, self.heads, self.head_dim

    @property
    def partial_lse_shape(self) -> tuple[int, int, int]:
        return self.num_splits, self.tokens, self.heads

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.tokens, self.heads, self.head_dim


@dataclass(frozen=True)
class NormRopeCase:
    tokens: int
    head_dim: int = 256
    q_heads: int = 16
    k_heads: int = 1
    rotary_dim: int = 16
    table_tokens: int = 4096

    @property
    def q_shape(self) -> tuple[int, int]:
        return self.tokens, self.q_heads * self.head_dim

    @property
    def k_z_shape(self) -> tuple[int, int]:
        return self.tokens, self.k_heads * self.head_dim

    @property
    def table_shape(self) -> tuple[int, int]:
        return self.table_tokens, self.rotary_dim


@dataclass(frozen=True)
class LogitsCase:
    seq_q: int = 8
    groups: int = 4
    heads_per_group: int = 4
    proxy_dim: int = 256
    seq_k: int = 64
    block_q: int = 64
    block_k: int = 128

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        return self.seq_q, self.groups, self.heads_per_group, self.proxy_dim

    @property
    def weights_shape(self) -> tuple[int, int, int]:
        return self.seq_q, self.groups, self.heads_per_group

    @property
    def k_shape(self) -> tuple[int, int, int]:
        return self.seq_k, 1, self.proxy_dim

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.groups * self.seq_q, self.seq_k


MERGE_PRIMARY_CASES = (MergeCase(num_splits=2), MergeCase(num_splits=4))
NORM_PRIMARY_TOKENS = (1, 16, 512, 4096, 8192, 32768)
LOGITS_PRIMARY = LogitsCase()

# SHA-256 values are copied from the six frozen single_kernels inputs.  A unit
# test makes accidental fixture edits fail loudly instead of silently moving a
# later stage's baseline.
FROZEN_FIXTURE_SHA256 = {
    "example1_merge_split_before.py": "36a12fe7823cb2e3a3786931e57f57a5828199675526c664934b24ea3761e40e",
    "example1_merge_split_after.py": "4d0fb0518ed54c931150e06d1f1feca16f40c062e42a2a58d45ee075bebafd6e",
    "example2_indexer_norm_rope_before.py": "eb60b102f977f4a8f842708e07810078ffe316bd9c3276459bfd981ddf9af5ee",
    "example2_indexer_norm_rope_after.py": "22ee156b8d1dbb45924288cf5f1cd74aadc862b30cb40407629117863bd0ba95",
    "example2_indexer_logits_before.py": "b7adde9599d646a16f88bc48996e15bded82b7040bb6e326241a2e136a813312",
    "example2_indexer_logits_after.py": "bd93df7b23a67585aad76654208b2522ee3e8fe220088b8b7f9a6030b3202f0c",
}

# These notes are intentionally data, not an executable optimization policy.
# A later rule must prove its own legality rather than treating after DSL as an
# oracle merely because it is in this fixture package.
TARGET_STRUCTURES = {
    "merge_split": {
        "before": "one program per (token, head), reduction over split",
        "after_target": "adjacent-head aggregation with a head-tail mask",
        "strict_oracle": "before + reference only",
    },
    "norm_rope": {
        "before": "one program per (token, head), BF16 round-trip before RoPE",
        "after_target": "persistent token tiles and head-independent vector reuse",
        "strict_oracle": "before + reference only",
    },
    "indexer_logits": {
        "before": "one program per (Q tile, K tile, provider group)",
        "after_target": "group loop inside a 2-D Q/K grid with one reused K tile",
        "strict_oracle": "before + reference only",
    },
}

NON_ORACLE_AFTER_DIFFERENCES = {
    "merge_split":
    "after fixes BLOCK_D=head_dim; non-power-of-two head_dim is not a legal default",
    "norm_rope":
    "after performs a full store then overwrites the rotary span; it is not a transactional store oracle",
    "indexer_logits":
    "after carries num_stages=2 in the group loop; it is a structure target, not a graph-rule requirement",
}

MINIMAL_NEGATIVE_VARIANTS = {
    "merge_split": "heads=1 or an unknown/non-contiguous stride must preserve the before grid",
    "norm_rope": "aliasing output or 2*rotary_dim > head_dim must reject the transformed path",
    "indexer_logits": "groups=1 and a cost-rejected tiny grid must preserve the three-dimensional baseline",
}

SMOKE_SUITE = (
    "pytest -q third_party/ascend/unittest/pytest_ut/test_task_0005_harness_unit.py third_party/ascend/unittest/pytest_ut/test_task_0005_grid_specialization.py",
    "lit -sv third_party/ascend/unittest/Conversion/General/TritonToGraph/task-0005-*.mlir",
    "TASK0005_ENABLE_E2E=1 pytest -q third_party/ascend/unittest/pytest_ut/test_task_0005_harness_e2e.py",
)


def fixture_path(operator: OperatorName, variant: VariantName) -> Path:
    names = {
        ("merge_split", "before"): "example1_merge_split_before.py",
        ("merge_split", "after"): "example1_merge_split_after.py",
        ("norm_rope", "before"): "example2_indexer_norm_rope_before.py",
        ("norm_rope", "after"): "example2_indexer_norm_rope_after.py",
        ("indexer_logits", "before"): "example2_indexer_logits_before.py",
        ("indexer_logits", "after"): "example2_indexer_logits_after.py",
    }
    return FIXTURE_ROOT / names[operator, variant]


def assert_fixture_integrity() -> None:
    for filename, expected in FROZEN_FIXTURE_SHA256.items():
        path = FIXTURE_ROOT / filename
        assert path.is_file(), f"missing frozen fixture: {path}"
        actual = sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (f"fixture drift for {filename}: expected {expected}, got {actual}")


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError(f"invalid ceil_div({value}, {divisor})")
    return (value + divisor - 1) // divisor


def expected_grid(
    operator: OperatorName,
    case: MergeCase | NormRopeCase | LogitsCase,
    *,
    specialization: str | None = None,
) -> tuple[int, ...]:
    """Return the before-kernel launch grid, not a proposed optimized grid."""
    if operator == "merge_split":
        assert isinstance(case, MergeCase)
        return case.tokens, case.heads
    if operator == "norm_rope":
        assert isinstance(case, NormRopeCase)
        if specialization not in ("q", "k"):
            raise ValueError("norm_rope grid requires specialization='q' or 'k'")
        return case.tokens, case.q_heads if specialization == "q" else case.k_heads
    if operator == "indexer_logits":
        assert isinstance(case, LogitsCase)
        return (
            ceil_div(case.seq_q, case.block_q),
            ceil_div(case.seq_k, case.block_k),
            case.groups,
        )
    raise ValueError(f"unknown operator {operator!r}")


def expected_program_count(
    operator: OperatorName,
    case: MergeCase | NormRopeCase | LogitsCase,
    *,
    specialization: str | None = None,
) -> int:
    grid = expected_grid(operator, case, specialization=specialization)
    product = 1
    for dimension in grid:
        product *= dimension
    return product


def case_payload(case: MergeCase | NormRopeCase | LogitsCase) -> dict[str, Any]:
    """Stable JSON payload with both scalar parameters and derived shapes."""
    payload = asdict(case)
    if isinstance(case, MergeCase):
        payload.update(
            partial_out_shape=list(case.partial_out_shape),
            partial_lse_shape=list(case.partial_lse_shape),
            output_shape=list(case.output_shape),
        )
    elif isinstance(case, NormRopeCase):
        payload.update(
            q_shape=list(case.q_shape),
            k_z_shape=list(case.k_z_shape),
            table_shape=list(case.table_shape),
        )
    else:
        payload.update(
            q_shape=list(case.q_shape),
            weights_shape=list(case.weights_shape),
            k_shape=list(case.k_shape),
            output_shape=list(case.output_shape),
        )
    return payload


def artifact_schema() -> dict[str, Any]:
    """Document the normalized Stage-00 artifact contract."""
    return {
        "schema_version":
        1,
        "required_files": (
            "environment.md",
            "correctness.json",
            "program_count.json",
            "performance.csv",
        ),
        "performance_columns": (
            "round",
            "variant",
            "operator",
            "case",
            "kernel_name",
            "source_csv",
            "source_row",
            "duration_us",
            "warmup_samples",
            "active_samples",
            "physical_npu",
            "compile_mode",
            "parallel_mode",
        ),
        "primary_compile_mode":
        COMPILE_MODE,
        "target_arch":
        TARGET_ARCH,
    }
