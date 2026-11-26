from dataclasses import dataclass
from enum import Enum


class ParameterClass(Enum):
    RuntimeEligible = "RuntimeEligible"
    ScheduleReusable = "ScheduleReusable"
    StaticRequired = "StaticRequired"
    Unknown = "Unknown"


@dataclass(frozen=True)
class ParameterDecision:
    position: int
    classification: ParameterClass
    reasons: tuple


@dataclass(frozen=True)
class FunctionPlan:
    fn: object
    dynamic: tuple
    local_lines: tuple
    helpers: tuple  # ((call line, column), FunctionPlan); one copy per call context


@dataclass(frozen=True)
class LaunchRecipe:
    tile: int
    extent: int
    bound: int | None  # External tile count, or None when the loop derives coverage.
    pointers: tuple
    stores: tuple
    rule: str = "MaskedGridStrideElementwiseV2"
    local_loop: bool = False
    program_offsets: tuple = ()  # (pointer position, tile-independent base expression)
    grid_start: tuple = ("program_id", 0)  # Original grid-stride expressions, before algebraic simplification.
    grid_step: tuple = ("num_programs", 0)

    @property
    def tiles(self):
        return (self.tile, )


@dataclass(frozen=True)
class TileAccess:
    pointer: int
    offset: tuple
    mask: tuple
    shape: tuple
    write: bool


@dataclass(frozen=True)
class LocalTileRecipe:
    tile: int
    extent: tuple
    accesses: tuple
    shapes: tuple
    rule: str = "MaskedLocalAxisV1"
    local_loop: bool = True
    axes: tuple = ()
    loops: tuple = ()
    index_checks: tuple = ()
    static_parameters: tuple = ()
    grid_axis: int | None = None
    outer_loops: tuple = ()
    grid_stride_axis: int | None = None
    partition_parameters: tuple = ()
    control_parameters: tuple = ()

    @property
    def tiles(self):
        return (self.tile, )


@dataclass(frozen=True)
class ComposedTileRecipe:
    """Independent parameterized axis proofs with deterministic candidate checks."""
    recipes: tuple
    rule: str = "IndependentTileAxesV2"

    @property
    def tiles(self):
        return tuple(recipe.tile for recipe in self.recipes)

    @property
    def static_parameters(self):
        return tuple(sorted(set().union(*(set(r.static_parameters) for r in self.recipes)) - set(self.tiles)))


@dataclass(frozen=True)
class AutoReuseProfile:
    source_key: str
    decisions: tuple
    plan: FunctionPlan
    recipe: object = None
    value_independent: tuple = ()  # Parameter values omitted from the analysis cache key.


@dataclass(frozen=True)
class BuildSpec:
    owner: object
    source: object
    memory_key: str
    variant_key: str
    family_key: tuple
    options: object
    specialization: tuple
    source_guard: object = None


@dataclass(frozen=True)
class ExecutableVariant:
    kernel: object
    source: object
    variant_key: str
    family_key: tuple
    load_context: tuple


@dataclass(frozen=True)
class PreparedLaunch:
    executable: ExecutableVariant
    arguments: tuple
    grid: tuple | None = None


@dataclass(frozen=True)
class JitTailRequest:
    desired: BuildSpec
