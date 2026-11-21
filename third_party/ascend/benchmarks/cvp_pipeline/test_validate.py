"""Collector provenance checks without importing Torch or starting hardware."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from validate import pinned_tools, sha, validate_frontend_input


class ValidatorProvenanceTest(unittest.TestCase):

    def test_direct_frontend_has_only_an_independent_stage_reader(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            frontend, reader, core = [root / name for name in ('frontend', 'reader', 'core.so')]
            for path in (frontend, reader, core):
                path.write_bytes(path.name.encode())
            args = SimpleNamespace(frontend=frontend, frontend_sha256=sha(frontend), reader=reader,
                                   reader_sha256=sha(reader), libtriton_sha256=None)
            compiler = SimpleNamespace(_get_npucompiler_path=lambda: (frontend, {}))
            selected, checker, identity = pinned_tools(compiler, SimpleNamespace(), SimpleNamespace(__file__=core),
                                                       args)
            self.assertEqual(selected, frontend.resolve())
            self.assertEqual(checker, reader.resolve())
            self.assertIsNone(identity['normal_compile_reader'])
            self.assertEqual(identity['frontend_input_stage'], 'ttadapter')
            self.assertEqual(identity['in_process_ir']['sha256'], sha(core))
            self.assertEqual(identity['stage_evidence_reader']['sha256'], sha(reader))
            args.reader_sha256 = 'wrong checker'
            with self.assertRaises(AssertionError):
                pinned_tools(compiler, SimpleNamespace(), SimpleNamespace(__file__=core), args)

    def test_captured_frontend_input_must_match_the_current_compiled_stage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / 'input.mlir'
            source.write_bytes(b'module {}\n')
            command = dict(input_path=str(source), input_sha256=sha(source))
            validate_frontend_input({'ttadapter': 'module {}\n'}, command, 'ttadapter')
            with self.assertRaises(AssertionError):
                validate_frontend_input({'ttadapter': 'different'}, command, 'ttadapter')
            command['input_sha256'] = 'wrong recorded bytes'
            with self.assertRaises(AssertionError):
                validate_frontend_input({'ttadapter': 'module {}\n'}, command, 'ttadapter')

    def test_help_describes_the_checker_without_initializing_hardware(self):
        result = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).with_name('validate.py')), '--help'], capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('independent CPU stage-evidence reader', result.stdout)
        self.assertIn('--libtriton-sha256', result.stdout)


if __name__ == '__main__':
    unittest.main()
