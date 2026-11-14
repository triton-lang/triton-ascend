# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import importlib
import subprocess
from types import SimpleNamespace

import pytest

from block_batching_contract_host import PACKAGE

vendor = importlib.import_module(PACKAGE + '.block_batching_vendor')
pytestmark = pytest.mark.backend('native')

OVERFLOW = b'ub overflow, requires 3072256 bits while 1572864 bits available!'
PIPELINE = b'loc("kernel.mlir":2:1): error: Failed to run BishengHIR pipeline\n'
OPTIONS = SimpleNamespace(block_batching_layout='auto', block_batching_factor=4, block_batching_axis=-1)
REWRITTEN = {'layout': {'accesses': [{'rewritten': True}]}}

# Captured from the Ascend 950 A5 regbase failure. Only the temporary path is normalized.
A5_LAYOUT = b'''[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
loc("/tmp/build/kernel.mlir":1:1): error: Failed to run buildFinalHIVMPipelines pipeline

loc("/tmp/build/kernel.mlir":2:3): error: ub overflow, requires 2097152 bits while 2031616 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
[ERROR] Failed to run BiShengIR regbase pipeline'''


def test_a5_vv_layout_budget_exhaustion():
    err = subprocess.CalledProcessError(1, ['vendor'], stderr=A5_LAYOUT)
    refusal = vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='disabled'), OPTIONS, cv_resources=True)
    assert (refusal.reason, refusal.stage, refusal.factor, refusal.axis) == ('LAYOUT_TARGET_RESOURCE_REJECTED',
                                                                             'npubin', 4, -1)
    assert refusal.diagnostic == A5_LAYOUT.decode()


@pytest.mark.parametrize('rc, stdout, stderr', [
    (-11, b'', A5_LAYOUT),
    (134, b'', A5_LAYOUT),
    (139, b'', A5_LAYOUT),
    (2, b'', A5_LAYOUT),
    (1, b'LLVM ERROR: verifier failed', A5_LAYOUT),
    (1, b'', A5_LAYOUT + b'\n[WARNING] unexpected subpass failure'),
    (1, b'', A5_LAYOUT.replace(b'[INFO]', b'[ERROR] LLVM ERROR:')),
    (1, b'', A5_LAYOUT.replace(b'2097152', b'2031616')),
    (1, b'', A5_LAYOUT.replace(b'ub overflow', b'ca overflow')),
    (1, b'', A5_LAYOUT.replace(b'kernel.mlir":2', b'other.mlir":2')),
])
def test_a5_mixed_or_unproven_failure_is_internal(rc, stdout, stderr):
    err = subprocess.CalledProcessError(rc, ['vendor'], output=stdout, stderr=stderr)
    assert vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='disabled'), OPTIONS, cv_resources=True) is None


@pytest.mark.parametrize('mode, rewritten, policy', [
    ('independent_batch_v1', True, 'auto'),
    ('spatial_concat_v1', True, 'auto'),
    ('disabled', False, 'auto'),
    ('disabled', True, 'semantic_only'),
])
def test_a5_transcript_only_refuses_rewritten_vv_auto_layout(mode, rewritten, policy):
    err = subprocess.CalledProcessError(1, ['vendor'], stderr=A5_LAYOUT)
    report = {'cv_mode': mode, 'layout': {'accesses': [{'rewritten': rewritten}]}}
    options = SimpleNamespace(block_batching_layout=policy, block_batching_factor=4, block_batching_axis=-1)
    for cv_resources in (True, False):
        assert vendor.classify_vendor_failure(err, report, options, cv_resources=cv_resources) is None


@pytest.mark.parametrize('rc, stdout, stderr', [
    (-11, b'', OVERFLOW),
    (134, OVERFLOW, b''),
    (139, b'', OVERFLOW),
    (0, b'', OVERFLOW),
    (1, b'LLVM ERROR: verifier failed', OVERFLOW),
    (1, OVERFLOW, b'unexpected failure'),
    (1, b'unsupported pass: insufficient UB', b''),
    (1, b'', b'ub overflow, requires 1 bits while 4 bits available!'),
    (1, b'', OVERFLOW + b' extra diagnostics'),
])
def test_unmodeled_vendor_exit_never_becomes_resource_decline(rc, stdout, stderr):
    err = subprocess.CalledProcessError(rc, ['vendor'], output=stdout, stderr=stderr)
    vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='independent_batch_v1'), OPTIONS, cv_resources=True)
    assert err.stage == 'npubin'
    assert vendor.resource_kind(rc, stdout, stderr) is None


@pytest.mark.parametrize('buf', [b'ub', b'cbuf', b'ca', b'cb', b'cc'])
def test_cv_takes_precedence_and_preserves_both_vendor_streams(buf):
    # The compiler can place its pipeline line and capacity error on different streams.
    stdout = PIPELINE
    stderr = b'loc("kernel.mlir":3:3): error: ' + OVERFLOW.replace(b'ub', buf)
    err = subprocess.CalledProcessError(1, ['vendor'], output=stdout, stderr=stderr)
    refusal = vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='independent_batch_v1'), OPTIONS,
                                             cv_resources=True)
    assert refusal.reason == 'CV_TARGET_RESOURCE_REJECTED' and refusal.stage == 'npubin'
    assert stdout.rstrip().decode() in refusal.diagnostic
    assert stderr.decode() in refusal.diagnostic


def test_only_rewritten_auto_layout_gets_ub_retry():
    err = subprocess.CalledProcessError(1, ['vendor'], output=OVERFLOW, stderr=b'')
    assert vendor.classify_vendor_failure(err, {}, OPTIONS, cv_resources=False) is None
    semantic = SimpleNamespace(block_batching_layout='semantic_only', block_batching_factor=4, block_batching_axis=-1)
    assert vendor.classify_vendor_failure(err, REWRITTEN, semantic, cv_resources=False) is None
    for cv_resources in (True, False):
        rejected = vendor.classify_vendor_failure(err, REWRITTEN, OPTIONS, cv_resources=cv_resources)
        assert (rejected.reason, rejected.stage, rejected.factor, rejected.axis) == ('LAYOUT_TARGET_RESOURCE_REJECTED',
                                                                                     'npubin', 4, -1)


def test_multiline_bisheng_capacity_message():
    diagnostic = PIPELINE + OVERFLOW + (
        b' (possible reason\nlarge or block number is more than what user expect '
        b'due to multi-buffer feature is enabled and some ops need extra local buffer. )')
    assert vendor.resource_kind(1, b'', diagnostic) == 'ub'


@pytest.mark.parametrize('empty', [True, False])
def test_success_without_candidate_object_is_internal(tmp_path, empty):
    error_type = importlib.import_module(PACKAGE + '.block_batching_worker').BatchCompileError
    output = tmp_path / 'kernel.o'
    if empty:
        output.touch()
    result = subprocess.CompletedProcess(['vendor'], 0, b'vendor stdout', OVERFLOW)
    with pytest.raises(error_type) as failed:
        vendor.require_candidate_binary(output, result)
    assert (failed.value.kind, failed.value.stage) == ('InvalidArtifact', 'npubin')
    assert (failed.value.stdout, failed.value.stderr) == (result.stdout, result.stderr)
    output.write_bytes(b'vendor payload')
    vendor.require_candidate_binary(output, result)
    assert output.read_bytes() == b'vendor payload'
