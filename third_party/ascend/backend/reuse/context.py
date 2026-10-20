from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_exact_request = ContextVar("ascend_reuse_exact_request", default=False)


@contextmanager
def exact_request_scope():
    token = _exact_request.set(True)
    try:
        yield
    finally:
        _exact_request.reset(token)


def requires_exact_request():
    return _exact_request.get()


def requires_sync_compilation():
    """One callback policy for tile reuse and its background compilation.

    Compilation listeners and load/launch hooks observe native foreground
    events and do not block reuse or worker compilation. Workers do not receive
    parent callbacks; consuming their cache entries follows the normal JIT tail.
    For compilation-altering hooks, match native activation: inspection uses
    presence and JIT hooks use truth testing.
    """
    from triton import knobs

    return (knobs.cache.manager_class is not None or knobs.cache.remote_manager_class is not None
            or knobs.runtime.add_stages_inspection_hook is not None or any(
                (knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook, knobs.compilation.override,
                 knobs.compilation.always_compile)))


@dataclass
class InvocationFrame:
    grid_resolved: bool = False
    grid: object = None
    launch_started: bool = False
    load_hook_started: bool = False
    compilation_listener_started: bool = False

    def resolve_grid(self, grid, bound_args):
        if not self.grid_resolved:
            self.grid = grid(bound_args) if callable(grid) else grid
            self.grid_resolved = True
        return self.grid

    def before_launch(self):
        self.launch_started = True

    def before_compilation_listener(self):
        self.compilation_listener_started = True

    def select_grid(self, grid):
        """Install a proved candidate grid only after dispatch has selected it."""
        if self.launch_started:
            raise RuntimeError("cannot change grid after launch")
        self.grid = grid
        self.grid_resolved = True
