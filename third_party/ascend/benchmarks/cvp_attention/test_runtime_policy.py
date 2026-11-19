"""CPU checks of metadata policy; automatic default still needs real JIT evidence."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from runtime_common import (compiler_tools, launch_options, sha_file,
                            validate_frontend_inputs, validate_native_route_metadata)


class CompilerProvenanceTest(unittest.TestCase):
    def test_direct_text_pins_loaded_library_without_a_reader_getter(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            site = Path(directory)
            frontend, core = site/'bishengir-compile', site/'libtriton.cpython-311.so'
            frontend.write_bytes(b'frontend')
            core.write_bytes(b'loaded extension')
            config = dict(wheel_site=str(site), frontend_sha256=sha_file(frontend),
                          libtriton_sha256=sha_file(core), reader_sha256='unused legacy config')
            compiler = SimpleNamespace(_get_npucompiler_path=lambda: (frontend, {}))
            tools = compiler_tools(compiler, SimpleNamespace(), SimpleNamespace(__file__=core), config)
            self.assertEqual(set(tools), {'frontend', 'libtriton'})
            self.assertEqual(tools['libtriton']['path'], str(core.resolve()))
            config['libtriton_sha256'] = 'incorrect'
            with self.assertRaises(AssertionError):
                compiler_tools(compiler, SimpleNamespace(), SimpleNamespace(__file__=core), config)

    def test_legacy_bridge_keeps_its_real_pinned_reader(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            site = Path(directory)
            frontend, core, reader = [site/name for name in ('frontend', 'core.so', 'reader')]
            for path in (frontend, core, reader):
                path.write_bytes(path.name.encode())
            config = dict(wheel_site=str(site), frontend_sha256=sha_file(frontend),
                          libtriton_sha256=sha_file(core), reader_sha256=sha_file(reader))
            compiler = SimpleNamespace(_get_npucompiler_path=lambda: (frontend, {}),
                                       bc_to_linalg_by_bishengir_opt=lambda: None)
            utils = SimpleNamespace(_get_bishengir_opt_path=lambda: (reader, {}))
            tools = compiler_tools(compiler, utils, SimpleNamespace(__file__=core), config)
            self.assertEqual(tools['reader']['sha256'], sha_file(reader))

    def test_actual_text_is_bound_to_its_jit_stage_without_synthetic_bytecode(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory)/'input.mlir'
            source.write_bytes(b'module {}\n')
            command = dict(input_path=str(source), input_sha256=sha_file(source))
            validate_frontend_inputs({'ttadapter': 'module {}\n'}, [command], 'ttadapter')
            with self.assertRaises(AssertionError):
                validate_frontend_inputs({'ttadapter': 'module { /* stale */ }\n'}, [command], 'ttadapter')
            source.write_bytes(b'changed after capture')
            with self.assertRaises(AssertionError):
                validate_frontend_inputs({'ttadapter': 'module {}\n'}, [command], 'ttadapter')

class NativeQueryPolicyTest(unittest.TestCase):
    def test_manifest_uses_one_common_candidate_and_no_removed_presets(self):
        recipes=json.loads(Path(__file__).with_name('recipes.json').read_text())
        self.assertEqual(set(recipes['presets']), {'baseline_hd64','baseline_hd128','common'})
        common=recipes['presets']['common']
        self.assertEqual(len(common),27)
        self.assertEqual(common['cvp_invariant_cache_mode'],'auto')
        self.assertEqual(common['cvp_reduction_chains'],2)
        self.assertTrue(all(c['initial_candidate']=='common' for c in recipes['cases']))
        production=[c for c in recipes['cases'] if c['shape'][:2]==[128,8]]
        self.assertEqual(len(production),11)
        for case in production:
            self.assertEqual(case['block_m'],128)
            self.assertEqual(case['block_n'],256 if case['shape'][-1]==64 else 128)
        self.assertIs(launch_options(common)['disable_tightly_coupled_buffer_reuse'],True)

    def test_removed_query_unroll_is_not_a_benchmark_route(self):
        for lanes in (1,2,4):
            with self.subTest(lanes=lanes),self.assertRaisesRegex(AssertionError,'Unroll was removed'):
                launch_options(dict(enable_dynamic_cv_pipeline=False,enable_preload=False,
                    enable_cvp_plan_analysis=False,enable_cvp_persistent_pipeline=True,
                    cvp_persistent_pipeline_mode='unroll',cvp_inner_unroll=lanes))

    def test_ordinary_default_remains_valid(self):
        validate_native_route_metadata(dict(ir_override=None,bitcodes=None))
        validate_native_route_metadata(dict(ir_override=None,bitcodes=[],enable_cvp_persistent_pipeline=False,
                                           disable_tightly_coupled_buffer_reuse=False))

    def test_query_requires_exact_effective_true(self):
        for value in (None,False,0,1,'true'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_native_route_metadata(dict(ir_override=None,bitcodes=[],enable_cvp_persistent_pipeline=True,
                                                   disable_tightly_coupled_buffer_reuse=value))

    def test_protected_query_allows_normal_empty_bitcode_defaults(self):
        for value in (None,[],()):
            validate_native_route_metadata(dict(ir_override=None,bitcodes=value,enable_cvp_persistent_pipeline=True,
                                               disable_tightly_coupled_buffer_reuse=True))

    def test_protected_policy_does_not_admit_injected_ir_or_bitcodes(self):
        for change in (dict(ir_override='file.mlir'),dict(bitcodes=['private.bc'])):
            metadata=dict(ir_override=None,bitcodes=[],enable_cvp_persistent_pipeline=True,
                          disable_tightly_coupled_buffer_reuse=True)
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_native_route_metadata(dict(metadata,**change))

if __name__=='__main__':unittest.main(verbosity=2)
