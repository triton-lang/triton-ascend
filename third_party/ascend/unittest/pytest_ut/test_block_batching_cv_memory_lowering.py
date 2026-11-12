"""The masked batch DMA must not shrink its allocated fractal footprint."""
import copy
from types import SimpleNamespace
import pytest
from triton.backends.ascend import compiler
from triton.backends.ascend.block_batching import BatchingRejected
from triton.backends.ascend.block_batching_guard_v2 import check_guard_v2

pytestmark = pytest.mark.backend("native")


@pytest.mark.parametrize("lowering", [None, "static_masked_lanes_v1", "spatial_m_v1", "rank3_full_tile_batch_v2"])
def test_other_lowerings_keep_default_memory_contract(lowering):
    metadata = {"block_batching": {"cv_lowering": lowering}}
    before = copy.deepcopy(metadata)
    assert compiler._masked_batch_compile_options(metadata, "/compiler") == []
    assert metadata == before


@pytest.mark.parametrize("arch,proof,accepted", [("Ascend950PR_9579", True, True), ("Ascend910B4", True, False),
                                                 ("Ascend950PR_9579", False, False), ("Ascend950PR_9579", 1, False)])
@pytest.mark.parametrize("lowering", ["rank3_masked_batch_v1", "rank3_uniform_control_batch_v1"])
def test_masked_backend_contract_requires_arch_and_padding_proof(arch, proof, accepted, lowering):
    report = dict(cv_lowering=lowering, guard=dict(require_same_fractal_bounds=proof))
    metadata = dict(block_batching=report, target=SimpleNamespace(arch=arch))
    if accepted:
        assert compiler._masked_batch_compile_options(metadata, "/compiler") == []
        assert report["cv_memory_lowering"] == "full_reduction_guard_v3"
    else:
        with pytest.raises(BatchingRejected, match="CV_MASKED_BATCH_MEMORY_LOWERING_UNSUPPORTED"):
            compiler._masked_batch_compile_options(metadata, "/compiler")


def matrix_guard(width=4, transpose=False):
    const = lambda n: ["const", n, 32]
    mul = lambda a, b: ["muli", 32, a, b]
    add = lambda a, b: ["addi", 32, a, b]
    row, col = ["coord", 0, 32], ["coord", 1, 32]
    index = add(mul(col, const(32)), row) if transpose else add(mul(row, const(32)), col)
    offset = mul(add(mul(["pid", 0], const(1024)), index), const(width))
    mask = ["and", ["slt", row, ["arg", 1, 32], 32], ["slt", col, ["arg", 2, 32], 32]]
    return dict(
        version=2, axis=0, require_same_fractal_bounds=True, accesses=[
            dict(root=0, shape=[32, 32], bytes=width, write=False, offset=offset, mask=mask, pointer_checks=[],
                 loops=[])
        ])


def test_reduction_axis_must_be_complete():
    g = matrix_guard()
    a = g['accesses'][0]
    from triton.backends.ascend.block_batching_guard_v2 import ByteDomain, GuardFailure
    domain = ByteDomain([4096, 32, 29], [8, 1, 1], a)
    with pytest.raises(GuardFailure, match='CV_BATCH_REDUCTION_MASK_NOT_PROVEN'):
        domain.require_full_axes([1])
    assert domain.require_full_axes([0]) is None


def test_invalid_dot_contract_fails_closed():
    g = matrix_guard()
    assert check_guard_v2(g, [4096, 32, 32], [8, 1, 1], 2) == (False, "CV_BATCH_DOT_CONTRACT_NOT_PROVEN")


def proof_only(guard, arguments, grid=(8, 1, 1)):
    from triton.backends.ascend.block_batching_guard_v2 import ByteDomain
    access = guard['accesses'][0]
    domain = ByteDomain(arguments, grid, access)
    try:
        domain.require_same_fractal_bounds()
    except Exception as e:
        return False, str(e)
    return True, 'PROVEN_AT_RUNTIME'


@pytest.mark.parametrize("width,transpose,m,n,accepted", [
    (4, False, 32, 32, True),
    (4, False, 31, 29, True),
    (4, False, 32, 25, True),
    (4, False, 32, 24, False),
    (4, False, 32, 23, False),
    (4, False, 17, 32, False),
    (4, False, 16, 32, False),
    (4, False, 0, 32, False),
    (4, False, 1, 1, False),
    (4, True, 29, 23, False),
    (4, True, 24, 32, False),
    (4, False, 25, 25, True),
    (4, True, 25, 25, True),
    (4, False, 24, 32, False),
    (4, True, 32, 24, False),
    (2, False, 23, 23, True),
    (2, False, 16, 32, False),
    (2, True, 23, 23, True),
])
def test_fractal_boundary_admission(width, transpose, m, n, accepted):
    result = proof_only(matrix_guard(width, transpose), [4096, m, n])
    assert result == ((True, "PROVEN_AT_RUNTIME") if accepted else (False, "CV_BATCH_PADDING_NOT_PROVEN"))


@pytest.mark.parametrize("mask", [["unknown"], ["false"], ["unsupported"]])
def test_unknown_mask_never_authorizes_padded_batch(mask):
    g = matrix_guard()
    g["accesses"][0]["mask"] = mask
    assert proof_only(g, [4096, 32, 32]) == (False, "CV_BATCH_PADDING_NOT_PROVEN")


def test_padding_proof_checks_every_original_program():
    g = matrix_guard()
    a = g["accesses"][0]
    a["mask"][1][1] = ["addi", 32, ["muli", 32, ["pid", 1], ["const", 32, 32]], ["coord", 0, 32]]
    assert proof_only(g, [4096, 57, 32], (8, 2, 1))[0]
    assert proof_only(g, [4096, 56, 32], (8, 2, 1)) == (False, "CV_BATCH_PADDING_NOT_PROVEN")


@pytest.mark.parametrize("value", [1, "true", None])
def test_padding_guard_flag_requires_boolean(value):
    g = matrix_guard()
    g["require_same_fractal_bounds"] = value
    assert check_guard_v2(g, [4096, 32, 32], [8, 1, 1], 2) == (False, "INVALID_GUARD_SCHEMA")


@pytest.mark.parametrize("m,k,reason", [
    (1, 1, "CV_BATCH_PADDING_NOT_PROVEN"),
    (24, 32, "CV_BATCH_PADDING_NOT_PROVEN"),
    (32, 29, "CV_BATCH_REDUCTION_MASK_NOT_PROVEN"),
    (29, 32, None),
    (32, 32, None),
])
def test_specialized_prefix_is_rejected_before_backend(m, k, reason):
    from triton.backends.ascend.block_batching_guard_v2 import static_cv_mask_refusal
    g = matrix_guard()
    g['cv_mask_contract'] = {'inputs': [{'access': 0, 'operand': 0, 'axes': [0, 1]}]}
    mask = g['accesses'][0]['mask']
    mask[1][2] = ['const', m, 32]
    mask[2][2] = ['const', k, 32]
    assert static_cv_mask_refusal(g) == reason


@pytest.mark.parametrize("bound", [['arg', 1, 32], ['grid', 0, 32], ['pid', 0, 32]])
def test_compile_time_mask_check_does_not_guess_dynamic_values(bound):
    from triton.backends.ascend.block_batching_guard_v2 import static_cv_mask_refusal
    g = matrix_guard()
    mask = g['accesses'][0]['mask']
    mask[1][2] = bound
    mask[2][2] = bound
    assert static_cv_mask_refusal(g) is None


def test_static_reduction_rejection_tracks_transposed_axis():
    from triton.backends.ascend.block_batching_guard_v2 import static_cv_mask_refusal
    g = matrix_guard(transpose=True)
    g['cv_mask_contract'] = {'inputs': [{'access': 0, 'operand': 0, 'axes': [1, 0]}]}
    mask = g['accesses'][0]['mask']
    mask[1][2] = ['const', 29, 32]
    assert static_cv_mask_refusal(g) == 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN'
