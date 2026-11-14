# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Classify the npubin compiler's explicit on-chip capacity diagnostics."""
import re
from pathlib import Path

from .block_batching import _is_cv_report
from .block_batching_worker import BatchDecline

# An MLIR location and severity may precede the actual vendor message. Keep
# this a whole diagnostic match: arbitrary text containing 'ub overflow' is
# not sufficient evidence for retrying compilation or selecting a baseline.
_PREFIX = r'(?:loc\([^\r\n]*?\):\s*)?(?:error:\s*)?'
_RESOURCE = re.compile(
    _PREFIX + r'(ub|cbuf|ca|cb|cc) overflow,\s*requires (\d{1,20}) bits while '
    r'(\d{1,20}) bits available[!.]?', re.I)
_PIPELINE = re.compile(_PREFIX + r'Failed to run BishengHIR pipeline[.!]?', re.I)
# Documented continuation of the vendor's UB message (Ascend programming guide).
_MULTIBUFFER_HINT = re.compile(
    r'\s*\(possible reason\s+large or block number is more than what user expect '
    r'due to multi-buffer feature is enabled and some ops need extra local buffer\.\s*\)', re.I)

# Ascend 950 A5 emits these together on stderr for a rejected regbase layout.
# Match the complete transcript: a pipeline summary alone is not a capacity error.
_A5_UB_DIAGNOSTIC = re.compile(
    r'\[INFO\] Using merged native A5 regbase pipeline\n'
    r'\[WARNING\] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer '
    r'and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1\.4\.0\. '
    r'Equivalent setting: --multibuffer-mode="\[\(gm, 2\), \(l1, 2\), \(l0c, 2\), \(ub, 2\)\]"\. '
    r'A3 default gm=4, A5 default gm=2; set a level to 1 to disable it\.\n'
    r"(?:warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'\.\n)+"
    r'loc\("(?P<file>[^"\r\n]+/kernel(?:\.ttadapter)?\.mlir)":\d+:\d+\): error: Failed to run buildFinalHIVMPipelines pipeline\n+'
    r'loc\("(?P=file)":\d+:\d+\): error: ub overflow, requires (?P<required>\d{1,20}) bits while '
    r'(?P<available>\d{1,20}) bits available! \(possible reason: tiling basic block is too large '
    r'or block number is more than what user expect due to multi-buffer feature is enabled and some ops '
    r'need extra local buffer\.\)\n\[ERROR\] Failed to run BiShengIR regbase pipeline')

_A3_UB_DIAGNOSTIC = re.compile(
    r'\[NOTE\] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation\.\n'
    r'\[NOTE\] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation\.\n'
    r'\[NOTE\] Due to Ub overflow, automatically set enable-code-motion to false, '
    r'set enable-auto-multi-buffer to false; compilation still failed\.\n'
    r'loc\("(?P<file>[^"\r\n]+/kernel(?:\.ttadapter)?\.mlir)":\d+:\d+\): error: Failed to run BiShengHIR pipeline\n+'
    r'loc\("(?P=file)":\d+:\d+\): error: ub overflow, requires (?P<required>\d{1,20}) bits while '
    r'(?P<available>\d{1,20}) bits available! \(possible reason: tiling basic block is too large '
    r'or block number is more than what user expect due to multi-buffer feature is enabled and some ops '
    r'need extra local buffer\.\)\n\[ERROR\] Failed to run BiShengIR pipeline')

# The A3 compiler may prepend an informational line to stderr and emit
# cv-pipelining warnings on stdout before the retry transcript.  Those lines
# do not change the resource diagnosis; unrelated errors must still prevent a
# fallback so that real compiler failures are not hidden.
_BENIGN_DIAGNOSTIC_LINE = re.compile(
    r'^(?:\s*|\[INFO\].*|\[WARNING\].*|warning:.*|'
    r'(?:loc\([^\r\n]*?\):\s*)+warning:.*)$', re.I)


def _benign_diagnostics(data):
    text = _text(data)
    return all(_BENIGN_DIAGNOSTIC_LINE.fullmatch(line) for line in text.splitlines() if line.strip())


def _a3_benign_diagnostics(data, expected_file):
    """Accept A3 warnings and the exact source-location echo for one retry."""
    text = _text(data)
    locations = re.compile(
        rf'^(?:loc\("{re.escape(expected_file)}":\d+:\d+\):\s*)+$')
    for line in text.splitlines():
        if not line.strip():
            continue
        if _BENIGN_DIAGNOSTIC_LINE.fullmatch(line) or locations.fullmatch(line):
            continue
        return False
    return True


def _a3_retry_match(stderr):
    """Match the complete A3 retry transcript with harmless compiler preamble."""
    text = _text(stderr)
    match = _A3_UB_DIAGNOSTIC.search(text)
    if match is None or int(match['required']) <= int(match['available']):
        return None
    if not _benign_diagnostics(text[:match.start()]) or not _benign_diagnostics(text[match.end():]):
        return None
    return match


def _text(data):
    return data.decode(errors='replace') if isinstance(data, bytes) else (data or '')


def resource_kind(returncode, stdout, stderr):
    """Return the buffer name only when all vendor diagnostics are accounted for."""
    if not 0 < returncode < 128:
        return None
    kind = None
    for stream in (stdout, stderr):
        for line in _MULTIBUFFER_HINT.sub('', _text(stream)).splitlines():
            line = line.strip()
            if not line or _PIPELINE.fullmatch(line):
                continue
            match = _RESOURCE.fullmatch(line)
            if match is None or int(match[2]) <= int(match[3]):
                return None
            buffer = match[1].lower()
            if kind is not None and kind != buffer:
                return None
            kind = buffer
    return kind


def diagnostic_text(stdout, stderr):
    return '\n'.join(filter(None, (_text(stdout), _text(stderr))))


def classify_vendor_failure(error, report, options, *, cv_resources):
    """Return a modeled refusal only for the matching candidate and npubin stage."""
    error.stage = 'npubin'
    kind = resource_kind(error.returncode, error.stdout, error.stderr)
    if kind is None and error.returncode == 1 and not _text(error.stdout).strip():
        match = _A5_UB_DIAGNOSTIC.fullmatch(_text(error.stderr).strip())
        if match and int(match['required']) > int(match['available']):
            kind = 'ub'
    if kind is None and error.returncode == 1:
        match = _a3_retry_match(error.stderr)
        if match and int(match['required']) > int(match['available']):
            # A3 can echo the same two locations on stdout without messages;
            # normal compiler warnings are also harmless.  Any non-benign
            # stdout (for example an LLVM error) remains an internal failure.
            if _a3_benign_diagnostics(error.stdout, match['file']):
                kind = 'ub'
    if kind is not None and cv_resources and _is_cv_report(report):
        return BatchDecline("CV_TARGET_RESOURCE_REJECTED", "npubin",
                            diagnostic=diagnostic_text(error.stdout, error.stderr)[-16000:])
    if options.block_batching_layout == 'auto' and any(
            row.get('rewritten') for row in report.get('layout', {}).get('accesses', ())):
        if kind == 'ub':
            return BatchDecline("LAYOUT_TARGET_RESOURCE_REJECTED", "npubin",
                                diagnostic=diagnostic_text(error.stdout, error.stderr)[-16000:],
                                factor=options.block_batching_factor, axis=options.block_batching_axis)
    if kind is not None and report.get('cv_mode') == 'disabled':
        return BatchDecline("VV_TARGET_RESOURCE_REJECTED", "npubin",
                            diagnostic=diagnostic_text(error.stdout, error.stderr)[-16000:])


def require_candidate_binary(path, result):
    """A zero exit needs a real, nonempty output before inspecting its metadata."""
    output = Path(path)
    if output.is_file() and output.stat().st_size > 0:
        return
    from .block_batching_worker import BatchCompileError
    diagnostic = diagnostic_text(result.stdout, result.stderr)[-16000:]
    error = BatchCompileError(f'Compiler exited successfully without a nonempty {output.name}\n{diagnostic}',
                              kind='InvalidArtifact', stage='npubin')
    error.stdout, error.stderr = result.stdout, result.stderr
    raise error
