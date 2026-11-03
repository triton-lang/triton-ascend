"""Host-only coverage; also runnable directly with Python, without Torch/NPU."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / "third_party/ascend/build/proton_patch.py"
SPEC = importlib.util.spec_from_file_location("ascend_proton_patch", HELPER)
proton_patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proton_patch)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class TestProtonPatch(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-proton-patch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init")
        self.git("config", "user.name", "Proton Test")
        self.git("config", "user.email", "proton-test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.original = self.root / "third_party/proton/original.txt"
        self.added = self.root / "third_party/proton/added.txt"
        self.original.parent.mkdir(parents=True)
        self.original.write_text("upstream\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "upstream")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.original.write_text("ascend\n", encoding="utf-8")
        self.added.write_text("new backend\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "source review")
        self.source = self.git("rev-parse", "HEAD").strip()
        self.git("switch", "-c", "proton_patch", self.base)
        proton_patch.export_patch(self.root, base=self.base, source=self.source)

    def git(self, *args):
        return proton_patch._git(self.root, *args).stdout.decode()

    def test_disabled_keeps_upstream_sources(self):
        self.assertFalse(proton_patch.set_proton_patch(False, self.root))
        self.assertEqual(self.original.read_text(), "upstream\n")
        self.assertFalse(self.added.exists())

    def test_enabled_is_idempotent(self):
        self.assertTrue(proton_patch.set_proton_patch(True, self.root))
        self.assertFalse(proton_patch.set_proton_patch(True, self.root))
        self.assertEqual(self.original.read_text(), "ascend\n")
        self.assertEqual(self.added.read_text(), "new backend\n")

    def test_on_off_on_cycle(self):
        proton_patch.set_proton_patch(True, self.root)
        self.assertTrue(proton_patch.set_proton_patch(False, self.root))
        self.assertFalse(proton_patch.set_proton_patch(False, self.root))
        self.assertFalse(self.added.exists())
        self.assertEqual(self.original.read_text(), "upstream\n")
        self.assertTrue(proton_patch.set_proton_patch(True, self.root))

    def test_apply_conflict_does_not_overwrite_edits(self):
        self.original.write_text("user edit\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "No files were changed"):
            proton_patch.set_proton_patch(True, self.root)
        self.assertEqual(self.original.read_text(), "user edit\n")
        self.assertFalse(self.added.exists())

    def test_restore_conflict_does_not_delete_user_file(self):
        proton_patch.set_proton_patch(True, self.root)
        self.added.write_text("user edit\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "No files were changed"):
            proton_patch.set_proton_patch(False, self.root)
        self.assertEqual(self.added.read_text(), "user edit\n")
        self.assertEqual(self.original.read_text(), "ascend\n")

    def test_existing_added_path_is_not_overwritten(self):
        self.added.write_text("user file\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            proton_patch.set_proton_patch(True, self.root)
        self.assertEqual(self.added.read_text(), "user file\n")
        self.assertEqual(self.original.read_text(), "upstream\n")

    def test_unrelated_edits_survive_apply_and_restore(self):
        unrelated = self.root / "third_party/proton/local.txt"
        unrelated.write_text("keep\n", encoding="utf-8")
        proton_patch.set_proton_patch(True, self.root)
        proton_patch.set_proton_patch(False, self.root)
        self.assertEqual(unrelated.read_text(), "keep\n")

    def test_patch_checksum_is_enforced(self):
        patch_file = self.root / proton_patch.PATCH_DIR / "proton.patch"
        patch_file.write_bytes(patch_file.read_bytes() + b"\n")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            proton_patch.set_proton_patch(True, self.root)
        self.assertEqual(self.original.read_text(), "upstream\n")

    def test_check_uses_an_independent_index(self):
        before = self.git("ls-files", "--stage")
        proton_patch.check_patch(self.root, source=self.source)
        self.assertEqual(before, self.git("ls-files", "--stage"))
        self.assertEqual(self.original.read_text(), "upstream\n")

    def test_check_detects_source_branch_drift(self):
        with self.assertRaisesRegex(RuntimeError, "source branch has changed"):
            proton_patch.check_patch(self.root, source=self.base)

    def test_check_rejects_staging_generated_sources(self):
        proton_patch.set_proton_patch(True, self.root)
        self.git("add", "third_party/proton")
        with self.assertRaisesRegex(RuntimeError, "do not commit generated patch changes"):
            proton_patch.check_patch(self.root)

    def test_export_refuses_patched_worktree(self):
        proton_patch.set_proton_patch(True, self.root)
        with self.assertRaisesRegex(RuntimeError, "Restore third_party/proton"):
            proton_patch.export_patch(self.root, source=self.source)

    def test_source_archive_does_not_require_git_metadata(self):
        with tempfile.TemporaryDirectory(prefix="proton-sdist-") as directory:
            root = Path(directory)
            shutil.copytree(self.root / "third_party", root / "third_party")
            self.assertFalse((root / ".git").exists())
            self.assertTrue(proton_patch.set_proton_patch(True, root))
            self.assertTrue(proton_patch.set_proton_patch(False, root))

    def test_build_hook_runs_before_cmake(self):
        setup_path = ROOT / "third_party/ascend/build/setup_patch.py"
        module = ast.parse(setup_path.read_text(encoding="utf-8"))
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "patch_module")
        build_class = next(node for node in function.body
                           if isinstance(node, ast.ClassDef) and node.name == "CMakeBuild")
        observed = []
        original = self.original

        class BaseBuild:

            def run(self):
                observed.append(original.read_text())

        namespace = {
            "_OrigCMakeBuild":
            BaseBuild,
            "_REPO_ROOT":
            self.root,
            "set_proton_patch":
            proton_patch.set_proton_patch,
            "_apply_triton_ascend_patch":
            lambda: None,
            "_clean_hitest_env":
            lambda: None,
            "os":
            os,
            "mod":
            SimpleNamespace(check_env_flag=lambda name, default: os.getenv(name, default).upper() in
                            ["ON", "1", "YES", "TRUE", "Y"]),
        }
        exec(compile(ast.Module(body=[build_class], type_ignores=[]), str(setup_path), "exec"), namespace)
        build = namespace["CMakeBuild"]()
        for flag in ("OFF", "ON", "ON", "OFF"):
            with patch.dict(os.environ, TRITON_BUILD_PROTON=flag, TRITON_ENABLE_COVERAGE_HITEST="0"):
                build.run()
        self.assertEqual(observed, ["upstream\n", "ascend\n", "ascend\n", "upstream\n"])


class TestRecordedProtonPatch(unittest.TestCase):

    def test_recorded_patch_matches_review_source(self):
        manifest = json.loads((ROOT / proton_patch.PATCH_DIR / "manifest.json").read_text())
        if subprocess.run(["git", "cat-file", "-e", manifest["source_commit"]], cwd=ROOT,
                          capture_output=True).returncode:
            self.skipTest("Source commit is not available in this checkout/archive")
        proton_patch.check_patch(ROOT)


if __name__ == "__main__":
    unittest.main()
