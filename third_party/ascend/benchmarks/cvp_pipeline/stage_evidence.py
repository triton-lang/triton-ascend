"""Certify stage contracts from IR captured during normal compilation.

The frontend suppresses successful MLIR remarks. Its pass dumps preserve the
actual operations instead; the pinned native reader independently analyzes and
verifies those operations. Captured IR is never a runtime compiler input.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

CAPTURE_FLAGS = (
    '--mlir-print-ir-before=hivm-verify-cvp-plan',
    '--mlir-print-ir-after=hivm-cvp-plan-prepare,hivm-verify-cvp-plan',
    '--mlir-print-ir-module-scope',
    '--mlir-disable-threading',
)
_HEADER = re.compile(
    r'^// -----// IR Dump (Before|After) [^\n]*'
    r'\((hivm-cvp-plan-prepare|hivm-verify-cvp-plan)\)'
    r"(?: \('func\.func' operation: @stage_chain\))? //----- //$",
    re.MULTILINE,
)
_PHASES = (
    ('After', 'hivm-cvp-plan-prepare'),
    ('Before', 'hivm-verify-cvp-plan'),
    ('After', 'hivm-verify-cvp-plan'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_snapshots(diagnostics):
    """Require one complete, unambiguous prepare/schedule/verify sequence."""
    headers = list(_HEADER.finditer(diagnostics))
    phases = tuple(match.groups() for match in headers)
    assert phases == _PHASES, ('ambiguous or missing pipeline snapshots', phases)
    snapshots = {}
    for index, (name, match) in enumerate(zip(('prepared', 'scheduled', 'verified'), headers)):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(diagnostics)
        section = diagnostics[match.end():end]
        module = re.search(r'^module(?:\s|\{)', section, re.MULTILINE)
        assert module is not None, ('missing module snapshot', name)
        # Module-scope printing indents nested regions; only the outer module
        # closes at column zero. The native reader also parses/verifies each
        # extracted module, so malformed/truncated text cannot certify a case.
        close = re.search(r'^}\s*$', section[module.start():], re.MULTILINE)
        assert close is not None, ('incomplete module snapshot', name)
        text = section[:module.start() + close.end()].strip() + '\n'
        assert re.search(r'func\.func\s+@stage_chain\(', text), name
        snapshots[name] = text
    return snapshots


def _read(reader, source, passes, folder):
    folder.mkdir()
    argv = [str(reader), str(source), *passes, '-o', str(folder / 'output.mlir')]
    result = subprocess.run(argv, capture_output=True)
    (folder / 'stdout.txt').write_bytes(result.stdout)
    (folder / 'stderr.txt').write_bytes(result.stderr)
    row = {'argv': argv, 'exit_code': result.returncode, 'reader_sha256': sha(reader), 'input_sha256': sha(source)}
    (folder / 'command.json').write_text(json.dumps(row, indent=2) + '\n')
    result.check_returncode()
    return row, (result.stdout + result.stderr).decode(errors='replace')


def certify_stage_plan(diagnostics, folder, reader, segments):
    folder = Path(folder)
    snapshots = pipeline_snapshots(diagnostics)
    paths = {}
    for name, text in snapshots.items():
        paths[name] = folder / (name + '.mlir')
        paths[name].write_text(text)
    recognition, remarks = _read(reader, paths['prepared'], ['--hivm-cvp-plan-prepare'], folder / 'reader-recognition')
    recognized = [int(value) for value in re.findall(r'CVP stage plan recognized: (\d+) Cube/Vector segments', remarks)]
    assert recognized == [segments], recognized
    verification, remarks = _read(reader, paths['scheduled'],
                                  ['--hivm-verify-cvp-plan=drop-annotations=false retain-provenance=true'],
                                  folder / 'reader-verification')
    verified = [int(value) for value in re.findall(r'CVP stage plan verified: (\d+) ordered Skew stages', remarks)]
    assert verified == [2 * segments], verified
    completed, _ = _read(reader, paths['verified'], [], folder / 'reader-completed')
    return {
        'recognized_segments': recognized, 'verified_stages': verified, 'origin':
        'actual normal-frontend module snapshots; pinned native analysis/verification', 'snapshots':
        {name: {'path': str(path), 'sha256': sha(path)}
         for name, path in paths.items()}, 'reader_commands': [recognition, verification, completed]
    }
