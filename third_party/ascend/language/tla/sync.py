"""Explicit Ascend synchronization: intra-core pipes and cross-core flags.

Pipelines are compile-time ``PIPE`` enum members, e.g. ``PIPE.MTE2``.
Vector supports S, V, MTE2 and MTE3; Cube supports S, M, MTE1, MTE2,
MTE3 and F (the fixpipe). ALL selects a full pipeline barrier.
Event IDs are integers in [0, 7]; cross-core flag IDs are in
[0, 15]. ``sync_all`` uses the default global synchronization flags (10-14).
IDs accept constants or runtime integer scalars; runtime values must obey
the same hardware bounds. ``pipe`` and block ``mode`` are compile-time enum
members; the buffer-sync ``mode`` argument is a compile-time bool.
Vector supports ``PIPE.S/V/MTE2/MTE3``; Cube supports
``PIPE.S/M/MTE1/MTE2/MTE3/F``. ``PIPE.ALL`` is reserved for ``barrier``.

``MODE.INTER``, ``MODE.SUBBLOCK`` and ``MODE.BLOCK`` emit logical
``hivm.hir.sync_block_set/wait`` operations. The backend selects the target
instructions using the mode and core configuration. ``MODE.INTRA`` instead
emits ``hivm.hir.sync_intra_block_set/wait`` for a single physical A5 counter;
it performs no implicit partner-counter access. ``mutex_lock`` and
``mutex_unlock`` emit A5 buffer-token operations whose pipeline identifies
the issuing hardware unit.
"""

import enum

from triton.language.core import builtin, _unwrap_if_constexpr, tensor, int64

__all__ = [
    "barrier", "set_flag", "wait_flag", "sync_all", "block_set_flag", "block_wait_flag", "mutex_lock", "mutex_unlock",
    "sub_block_id", "MODE", "PIPE"
]

# Hardware ID bounds. The C++ builder further narrows the INTRA upper bound
# to 15 when the enclosing core is Vector.
MAX_EVENT_ID = 7
MAX_BLOCK_FLAG_ID = 15
MAX_INTRA_FLAG_ID = 31
MAX_BUFFER_ID = 31


class MODE(enum.Enum):
    """Compile-time mode for :func:`block_set_flag` and :func:`block_wait_flag`.

    ``INTER``, ``SUBBLOCK`` and ``BLOCK`` select logical inter-block,
    inter-subblock and intra-block synchronization respectively. The backend
    chooses the instruction sequence for the target and core ratio.

    ``INTRA`` selects a single physical Ascend950 counter. Vector subcores
    participate independently; Cube does not automatically access the partner
    counter at ``flagid + 16``. It is not interchangeable with ``BLOCK``.
    """

    INTER = 0
    SUBBLOCK = 1
    BLOCK = 2
    INTRA = 3


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


def _integer(value, name, max_value):
    """Check a compile-time integer lies in the hardware range [0, max_value]."""
    value = _unwrap_if_constexpr(value)
    if type(value) is not int:
        raise TypeError(f"{name} must be a compile-time integer")
    if not 0 <= value <= max_value:
        raise ValueError(f"{name} must be in [0, {max_value}], got {value}")
    return value


def _id(value, name, max_value, semantic):
    """Preserve static IDs; normalize runtime scalar integer IDs to i64.

    Runtime values must obey the same hardware bounds as constants. No
    runtime range check or clamping is inserted into synchronization code.
    """
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tensor):
        if value.type.is_block() or not value.dtype.is_int() or value.dtype.is_bool():
            raise TypeError(f"{name} must be an integer scalar (not bool)")
        return semantic.cast(value, int64).handle
    return _integer(value, name, max_value)


def _mode(mode):
    """Convert a compile-time block mode enum to its backend value."""
    mode = _unwrap_if_constexpr(mode)
    if not isinstance(mode, MODE):
        raise TypeError("mode must be a compile-time tla.sync.MODE enum member")
    return mode.value


@builtin
def barrier(pipe, _semantic=None):
    """Wait for all preceding instructions on a pipeline."""
    _semantic.builder.create_tla_barrier(_pipe(pipe))


@builtin
def set_flag(pipe1, pipe2, event_id, _semantic=None):
    """Signal completion from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_set_flag(_pipe(pipe1), _pipe(pipe2), _id(event_id, "event_id", MAX_EVENT_ID,
                                                                          _semantic))


@builtin
def wait_flag(pipe1, pipe2, event_id, _semantic=None):
    """Consume the event sent from producer ``pipe1`` to consumer ``pipe2``."""
    _semantic.builder.create_tla_wait_flag(_pipe(pipe1), _pipe(pipe2), _id(event_id, "event_id", MAX_EVENT_ID,
                                                                           _semantic))


@builtin
def sync_all(mode, _semantic=None):
    """Hardware global barrier: False synchronizes AIV, True all AIC/AIV.

    The frontend expands this to a pipeline barrier and block set/wait flags
    before handing IR to the backend. Call outside SIMD/SIMT scopes. In a mixed
    kernel, True outside core scopes emits both sides of the AIC/AIV handshake.
    """
    mode = _unwrap_if_constexpr(mode)
    if type(mode) is not bool:
        raise TypeError("sync_all mode must be a compile-time bool")
    _semantic.builder.create_tla_sync_all(mode)


@builtin
def block_set_flag(mode, pipe, flagid, _semantic=None):
    """Signal a logical cross-core flag or a physical intra-block counter.

    Logical modes lower to ``hivm.hir.sync_block_set``. ``MODE.INTRA`` lowers
    to ``hivm.hir.sync_intra_block_set`` and addresses exactly one counter.
    The enclosing ``tla.scope.cube()`` or ``tla.scope.vector()`` determines
    the core; outside an explicit core scope the builder defaults to Vector.

    :param mode: Compile-time synchronization mode.
    :type mode: MODE
    :param pipe: Producer pipeline. Vector accepts ``S``, ``V``, ``MTE2`` and
        ``MTE3``; Cube accepts ``S``, ``M``, ``MTE1``, ``MTE2``, ``MTE3`` and
        ``F``. ``ALL`` is not accepted.
    :type pipe: PIPE
    :param flagid: Constant or runtime scalar integer ID. Logical modes
        accept 0..15. ``INTRA`` accepts 0..15 on Vector and 0..31 on Cube.
        Boolean and block-tensor IDs are rejected.
    :type flagid: int or triton.language.tensor
    :return: ``None``; emits a synchronization operation.
    :rtype: None

    .. note::
        Runtime IDs are cast to ``int64`` without range checks or clamping;
        the caller must keep them within the same bounds as constant IDs.
        With ``INTRA``, explicitly arrange the matching peer/subcore wait;
        no partner access at ``flagid + 16`` is added automatically.
    """
    mode = _mode(mode)
    max_flag_id = MAX_INTRA_FLAG_ID if mode == MODE.INTRA.value else MAX_BLOCK_FLAG_ID
    _semantic.builder.create_tla_block_set_flag(mode, _pipe(pipe), _id(flagid, "flagid", max_flag_id, _semantic))


@builtin
def block_wait_flag(mode, pipe, flagid, _semantic=None):
    """Wait for a logical cross-core flag or a physical intra-block counter.

    Logical modes lower to ``hivm.hir.sync_block_wait``; ``MODE.INTRA``
    lowers to ``hivm.hir.sync_intra_block_wait``. Use the same synchronization
    protocol and mode as the matching :func:`block_set_flag`.

    :param mode: Compile-time synchronization mode.
    :type mode: MODE
    :param pipe: Consumer pipeline, subject to the same enclosing-core
        restrictions as :func:`block_set_flag`. ``ALL`` is not accepted.
    :type pipe: PIPE
    :param flagid: Constant or runtime scalar integer ID. Logical modes
        accept 0..15; ``INTRA`` accepts 0..15 on Vector and 0..31 on Cube.
        Boolean and block-tensor IDs are rejected.
    :type flagid: int or triton.language.tensor
    :return: ``None``; emits a synchronization operation.
    :rtype: None

    .. note::
        Runtime IDs are cast to ``int64`` without runtime bounds checks.
        ``INTRA`` waits on one physical counter and does not implicitly
        access ``flagid + 16`` on Cube.
    """
    mode = _mode(mode)
    max_flag_id = MAX_INTRA_FLAG_ID if mode == MODE.INTRA.value else MAX_BLOCK_FLAG_ID
    _semantic.builder.create_tla_block_wait_flag(mode, _pipe(pipe), _id(flagid, "flagid", max_flag_id, _semantic))


def _buffer_sync_args(bufid, pipe, mode, semantic):
    bufid = _id(bufid, "bufid", MAX_BUFFER_ID, semantic)
    pipe = _pipe(pipe)
    nonblocking = _unwrap_if_constexpr(mode)
    if type(nonblocking) is not bool:
        raise TypeError("mode must be a compile-time bool")
    return bufid, pipe, nonblocking


@builtin
def mutex_lock(bufid, pipe, mode=False, _semantic=None):
    """Acquire an Ascend950 buffer synchronization token (``hivm.hir.get_buf``).

    This is a buffer-ID-based synchronization instruction. The backend
    operation carries the pipeline, buffer ID and nonblocking flag; it has no
    separate core attribute. The frontend checks that ``pipe`` is valid for the
    enclosing core scope, defaulting to Vector when no scope is present.

    :param bufid: Token ID in 0..31, as a constant or runtime scalar integer.
        Boolean and block-tensor IDs are rejected.
    :type bufid: int or triton.language.tensor
    :param pipe: Compile-time issuing pipeline. Vector accepts ``S``, ``V``,
        ``MTE2`` and ``MTE3``; Cube additionally accepts ``M``, ``MTE1`` and
        ``F`` but not ``V``. ``ALL`` is not accepted.
    :type pipe: PIPE
    :param mode: Compile-time boolean; ``False`` blocks the issuing pipeline,
        while ``True`` selects nonblocking execution. Defaults to ``False``.
    :type mode: bool
    :return: ``None``; emits a token-acquire operation, not a success flag.
    :rtype: None

    .. note::
        Runtime IDs are cast to ``int64`` without bounds checks or clamping.
        This is a hardware pipeline token, not a host-side Python mutex.
    """
    _semantic.builder.create_tla_mutex_lock(*_buffer_sync_args(bufid, pipe, mode, _semantic))


@builtin
def mutex_unlock(bufid, pipe, mode=False, _semantic=None):
    """Release an Ascend950 buffer token (``hivm.hir.rls_buf``).

    :param bufid: Token ID in 0..31, as a constant or runtime scalar integer.
        Boolean and block-tensor IDs are rejected.
    :type bufid: int or triton.language.tensor
    :param pipe: Compile-time issuing pipeline, with the same enclosing-core
        restrictions as :func:`mutex_lock`. ``ALL`` is not accepted.
    :type pipe: PIPE
    :param mode: Compile-time boolean. ``False`` selects blocking execution;
        ``True`` selects nonblocking execution. Defaults to ``False``.
    :type mode: bool
    :return: ``None``; emits a token-release operation.
    :rtype: None

    .. note::
        Runtime IDs must remain in 0..31; no runtime bounds checks are
        inserted. Arrange the release as part of the token protocol used by
        :func:`mutex_lock`.
    """
    _semantic.builder.create_tla_mutex_unlock(*_buffer_sync_args(bufid, pipe, mode, _semantic))


@builtin
def sub_block_id(_semantic=None):
    """Index of the current AIV sub-block within its associated Cube block."""
    return tensor(_semantic.builder.create_tla_get_sub_block_idx(), int64)
