"""Explicit Ascend synchronization: intra-core pipes and cross-core flags.

Pipelines are compile-time ``PIPE`` enum members, e.g. ``PIPE.MTE2``.
Vector supports S, V, MTE2 and MTE3; Cube supports S, M, MTE1, MTE2,
MTE3 and F (the fixpipe). ALL selects a full pipeline barrier.
Event IDs are integers in [0, 7]; cross-core flag IDs are in
[0, 15]. ``sync_all`` uses the backend's default global synchronization flags.
IDs accept constants or runtime integer scalars; runtime values must obey
the same hardware bounds. Pipe and mode arguments remain compile-time values.

Cross-core flags are emitted as ``hivm.hir.sync_block_set/wait`` with an
explicit synchronization mode, which bishengir lowers to the FFTS
``SET.CROSS.CORE`` / ``WAIT.FLAG.DEV`` instructions.
"""

import enum

from triton.language.core import builtin, _unwrap_if_constexpr, tensor, int64

__all__ = [
    "barrier", "set_flag", "wait_flag", "sync_all", "block_set_flag", "block_wait_flag", "sub_block_id", "MODE", "PIPE"
]


class MODE(enum.Enum):
    """FFTS block synchronization mode for :func:`block_set_flag`."""

    INTER = 0
    SUBBLOCK = 1
    BLOCK = 2


class PIPE(enum.Enum):
    """Compile-time synchronization pipelines shared by Vector and Cube.

    ``F`` maps to the backend's ``PIPE_FIX``. ``PIPE_*`` spellings are aliases
    for callers using hardware-style names (for example, ``PIPE.PIPE_S``).
    """

    S = "PIPE_S"
    V = "PIPE_V"
    M = "PIPE_M"
    MTE1 = "PIPE_MTE1"
    MTE2 = "PIPE_MTE2"
    MTE3 = "PIPE_MTE3"
    F = "PIPE_FIX"
    ALL = "PIPE_ALL"

    PIPE_S = S
    PIPE_V = V
    PIPE_M = M
    PIPE_MTE1 = MTE1
    PIPE_MTE2 = MTE2
    PIPE_MTE3 = MTE3
    PIPE_FIX = F
    PIPE_ALL = ALL


def _pipe(pipe):
    """Convert a compile-time pipeline enum to its backend spelling."""
    pipe = _unwrap_if_constexpr(pipe)
    if not isinstance(pipe, PIPE):
        raise TypeError("pipe must be a compile-time tla.sync.PIPE enum member")
    return pipe.value


def _integer(value, name, maximum):
    """Check a compile-time integer lies in the hardware range [0, maximum]."""
    value = _unwrap_if_constexpr(value)
    if type(value) is not int:
        raise TypeError(f"{name} must be a compile-time integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _id(value, name, maximum, semantic):
    """Preserve static IDs; normalize runtime scalar integer IDs to i64.

    Runtime values must obey the same hardware bounds as constants. No
    runtime range check or clamping is inserted into synchronization code.
    """
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tensor):
        if value.type.is_block() or not value.dtype.is_int() or value.dtype.is_bool():
            raise TypeError(f"{name} must be an integer scalar (not bool)")
        return semantic.cast(value, int64).handle
    return _integer(value, name, maximum)


def _mode(mode):
    """Normalize a compile-time FFTS block mode to its enum value."""
    mode = _unwrap_if_constexpr(mode)
    if isinstance(mode, MODE):
        return mode.value
    if isinstance(mode, str):
        try:
            return MODE[mode.upper()].value
        except KeyError:
            raise ValueError(f"unknown block synchronization mode: {mode}") from None
    return _integer(mode, "mode", 2)


@builtin
def barrier(pipe, _semantic=None):
    """Wait for all preceding instructions on a pipeline."""
    _semantic.builder.create_tla_barrier(_pipe(pipe))


@builtin
def set_flag(pipe1, pipe2, event_id, _semantic=None):
    """Signal completion from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_set_flag(_pipe(pipe1), _pipe(pipe2), _id(event_id, "event_id", 7, _semantic))


@builtin
def wait_flag(pipe1, pipe2, event_id, _semantic=None):
    """Consume the event sent from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_wait_flag(_pipe(pipe1), _pipe(pipe2), _id(event_id, "event_id", 7, _semantic))


@builtin
def sync_all(mode, _semantic=None):
    """Hardware global barrier: False synchronizes AIV, True all AIC/AIV."""
    mode = _unwrap_if_constexpr(mode)
    if type(mode) is not bool:
        raise TypeError("sync_all mode must be a compile-time bool")
    _semantic.builder.create_tla_sync_all(mode)


@builtin
def block_set_flag(mode, pipe, flagid, _semantic=None):
    """
    Signal a cross-core FFTS flag from the current core.

    :param mode: ``MODE.INTER``, ``MODE.SUBBLOCK``, or ``MODE.BLOCK``
    :param pipe: signalling pipeline
    :param flagid: flag ID in [0, 15]
    """
    _semantic.builder.create_tla_block_set_flag(_mode(mode), _pipe(pipe), _integer(flagid, "flagid", 15))


@builtin
def block_wait_flag(mode, pipe, flagid, _semantic=None):
    """
    Wait for a cross-core FFTS flag on the given pipeline.

    :param mode: same FFTS mode as the matching :func:`block_set_flag`
    :param pipe: waiting pipeline
    :param flagid: flag ID in [0, 15], matching :func:`block_set_flag`
    """
    _semantic.builder.create_tla_block_wait_flag(_mode(mode), _pipe(pipe), _integer(flagid, "flagid", 15))


@builtin
def sub_block_id(_semantic=None):
    """Index of the current AIV sub-block within its associated Cube block."""
    return tensor(_semantic.builder.create_tla_get_sub_block_idx(), int64)
