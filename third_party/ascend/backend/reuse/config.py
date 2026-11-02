import os
from functools import lru_cache

SCHEMA_VERSION = 3  # Frozen compile request format.
RULE_VERSION = 20
ABI_VERSION = 2
ANALYSIS_NODE_BUDGET = 12000


@lru_cache(maxsize=1)
def mode():
    """Startup mode: 0 off, 1 dynamic constexpr, 2 dynamic constexpr and tile reuse."""
    try:
        value = int(os.environ.get("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", "0"))
    except ValueError:
        return 0
    return value if value in (0, 1, 2) else 0


def enabled():
    return mode() != 0


def supports_target(target):
    # D uses unannotated runtime arguments; the backend validates target/options.
    return target.backend == "npu"


def supports_tile_reuse(target, options):
    # The backend validates modes; pure SIMT has a separate execution contract.
    return mode() == 2 and supports_target(target) and options.compile_mode != "simt_only"
