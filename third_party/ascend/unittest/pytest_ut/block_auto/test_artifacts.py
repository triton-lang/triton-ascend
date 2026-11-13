"""CPU-only cache identity and same-compilation evidence regression tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

BACKEND = Path(__file__).resolve().parents[3] / "backend"
SPEC = importlib.util.spec_from_file_location("block_auto_test_artifacts", BACKEND / "block_auto_artifacts.py")
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class Cache:

    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def put(self, data, name, binary=True):
        path = self.directory / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return str(path)

    def get_file(self, name):
        path = self.directory / name
        return str(path) if path.is_file() else None


class ArtifactTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.temps = self.directory / "compiler-temporaries"
        self.temps.mkdir()
        (self.temps / "module.hivm.opt.mlir").write_text("final allocated IR")
        self.compiler = self.directory / "compiler"
        self.compiler.write_bytes(b"pinned compiler")
        self.binary = b"\x7fELFcandidate"
        self.asm = dict(ttir="tt.func public @kernel(%arg0: !tt.ptr<f32>) {}", ttadapter="adapter IR")
        self.metadata = dict(target=SimpleNamespace(arch="Ascend950PR_9579"), target_arch="Ascend950PR_9579",
                             block_auto_ttir_sha256=artifacts.digest(self.asm["ttir"].encode()),
                             block_auto_ttadapter_sha256=artifacts.digest(self.asm["ttadapter"].encode()))

    def cache(self, key):
        return Cache(self.directory / "cache" / key)

    def collect(self):
        evidence = artifacts.collect(self.temps, self.binary, self.compiler, artifacts.file_digest(self.compiler),
                                     self.metadata, ["compiler", "--simt-vf-dynamic-size=216"],
                                     cache_factory=self.cache)
        self.metadata["block_auto_evidence"] = evidence
        return evidence

    def kernel(self):
        return SimpleNamespace(kernel=self.binary, asm=self.asm,
                               metadata=SimpleNamespace(_asdict=lambda: dict(self.metadata)))

    def load(self):
        return artifacts.load_evidence(self.kernel(), cache_factory=self.cache, compiler_path=self.compiler)

    def test_split_cv_modules_survive_compiler_temporary_removal(self):
        for role in ("aic", "aiv"):
            (self.temps / ("kernel_mix_" + role + ".ll")).write_text(role + " LLVM")
        evidence = self.collect()
        self.assertEqual(evidence["status"], "COLLECTED")
        self.assertEqual(set(evidence["files"]), {"final_ir", "llvm_aic", "llvm_aiv"})
        for path in self.temps.iterdir():
            path.unlink()
        loaded = self.load()
        self.assertEqual(loaded["llvm_aic"], "aic LLVM")
        self.assertEqual(loaded["metadata"]["target"]["arch"], "Ascend950PR_9579")
        self.assertEqual(loaded["command"][-1], "--simt-vf-dynamic-size=216")
        self.assertTrue(Path(loaded["files"]["binary"]["path"]).is_file())

    def test_single_vv_or_cube_module_is_preserved_without_role_guessing(self):
        for text in ("AIV MIX LLVM", 'attributes #0 = { "cube-only-kernel" }'):
            with self.subTest(text=text):
                (self.temps / "kernel.ll").write_text(text)
                self.collect()
                self.assertEqual(self.load()["llvm"], text)

    def test_missing_llvm_disables_model_evidence(self):
        result = self.collect()
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("LLVM", result["reason"])

    def test_binary_rewrite_after_collection_is_not_silently_accepted(self):
        (self.temps / "kernel.ll").write_text("LLVM")
        self.collect()
        self.binary += b"changed"
        with self.assertRaisesRegex(ValueError, "binary changed"):
            self.load()

    def test_compiler_replacement_invalidates_cached_evidence(self):
        (self.temps / "kernel.ll").write_text("LLVM")
        self.collect()
        self.compiler.write_bytes(b"different compiler")
        with self.assertRaisesRegex(ValueError, "compiler identity"):
            self.load()

    def test_cached_llvm_corruption_is_detected(self):
        (self.temps / "kernel.ll").write_text("LLVM")
        evidence = self.collect()
        cache = self.cache(evidence["cache_key"])
        cache.put(b"corrupted", evidence["files"]["llvm"]["name"])
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.load()

    def test_frontend_stage_cannot_be_substituted_for_another_binary(self):
        (self.temps / "kernel.ll").write_text("LLVM")
        self.collect()
        self.asm["ttir"] = "other kernel IR"
        with self.assertRaisesRegex(ValueError, "frontend IR identity"):
            self.load()


if __name__ == "__main__":
    unittest.main()
