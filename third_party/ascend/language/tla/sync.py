"""Explicit Ascend synchronization: intra-core pipes and cross-core flags.

Pipelines are compile-time strings, e.g. ``"V"`` or ``"PIPE_MTE2"``.
Event IDs are compile-time integers in [0, 7]; cross-core flag IDs are in
[0, 15]. ``sync_all`` uses the backend's default global synchronization flags.

Cross-core flags are emitted as ``hivm.hir.sync_block_set/wait`` with an
explicit synchronization mode, which bishengir lowers to the FFTS
``SET.CROSS.CORE`` / ``WAIT.FLAG.DEV`` instructions. The A5 per-vector-core
``block_inter_*`` pair is not exposed yet, and the buffer-token instructions
(``get_buf``/``rls_buf``) have no backend operation.
"""

from triton.language.core import builtin, _unwrap_if_constexpr

__all__ = ["barrier", "set_flag", "wait_flag", "sync_all", "block_set_flag", "block_wait_flag"]

_PIPES = frozenset([
    "PIPE_S",
    "PIPE_V",
    "PIPE_M",
    "PIPE_MTE1",
    "PIPE_MTE2",
    "PIPE_MTE3",
    "PIPE_ALL",
    "PIPE_MTE4",
    "PIPE_MTE5",
    "PIPE_V2",
    "PIPE_FIX",
])


def _pipe(pipe):
    """Normalize a compile-time pipeline name and check it is known."""
    pipe = _unwrap_if_constexpr(pipe)
    if not isinstance(pipe, str):
        raise TypeError("pipe must be a compile-time string")
    pipe = pipe.upper()
    pipe = pipe if pipe.startswith("PIPE_") else "PIPE_" + pipe
    if pipe not in _PIPES:
        raise ValueError(f"unknown synchronization pipe: {pipe}")
    return pipe


def _integer(value, name, maximum):
    """Check a compile-time integer lies in the hardware range [0, maximum]."""
    value = _unwrap_if_constexpr(value)
    if type(value) is not int:
        raise TypeError(f"{name} must be a compile-time integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


@builtin
def barrier(pipe, _semantic=None):
    """Wait for all preceding instructions on a pipeline."""
    _semantic.builder.create_tla_barrier(_pipe(pipe))


@builtin
def set_flag(pipe1, pipe2, event_id, _semantic=None):
    """Signal completion from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_set_flag(_pipe(pipe1), _pipe(pipe2), _integer(event_id, "event_id", 7))


@builtin
def wait_flag(pipe1, pipe2, event_id, _semantic=None):
    """Consume the event sent from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_wait_flag(_pipe(pipe1), _pipe(pipe2), _integer(event_id, "event_id", 7))


@builtin
def sync_all(mode, _semantic=None):
    """Hardware global barrier: False synchronizes AIV, True all AIC/AIV."""
    mode = _unwrap_if_constexpr(mode)
    if type(mode) is not bool:
        raise TypeError("sync_all mode must be a compile-time bool")
    _semantic.builder.create_tla_sync_all(mode)


@builtin
def block_set_flag(mode, pipe, flag, _semantic=None):
    """
    Signal a cross-core flag from the current core.

    :param mode: 0 = inter-block, 1 = inter-subblock, 2 = intra-block
    :param pipe: signalling pipeline
    :param flag: flag ID in [0, 15]
    """
    _semantic.builder.create_tla_block_set_flag(_integer(mode, "mode", 2), _pipe(pipe), _integer(flag, "flag", 15))


@builtin
def block_wait_flag(pipe, flag, _semantic=None):
    """
    Wait for a cross-core flag on the given pipeline.

    :param pipe: waiting pipeline
    :param flag: flag ID in [0, 15], matching :func:`block_set_flag`
    """
    _semantic.builder.create_tla_block_wait_flag(_pipe(pipe), _integer(flag, "flag", 15))
