"""Host-only tests for the Proton build patch; run directly without Torch/NPU."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SETUP_PATH = ROOT / "third_party/ascend/build/setup_patch.py"
SETUP_AST = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
PATCH_PATH = Path("third_party/ascend/patch/proton_patch.patch")
TEST_PATCH = """diff --git a/third_party/proton/original.txt b/third_party/proton/original.txt
--- a/third_party/proton/original.txt
+++ b/third_party/proton/original.txt
@@ -1 +1 @@
-upstream
+ascend
diff --git a/third_party/proton/added.txt b/third_party/proton/added.txt
new file mode 100644
--- /dev/null
+++ b/third_party/proton/added.txt
@@ -0,0 +1 @@
+new backend
"""


def load_patch_functions(root):
    functions = [
        node for node in SETUP_AST.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_apply_patch", "_apply_proton_patch")
    ]
    namespace = {"_REPO_ROOT": root, "subprocess": subprocess}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SETUP_PATH), "exec"), namespace)
    return namespace


@unittest.skipUnless(shutil.which("git"), "Git is required")
class TestProtonPatch(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="test-proton-patch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "third_party/proton/original.txt"
        self.added = self.original.with_name("added.txt")
        self.original.parent.mkdir(parents=True)
        self.original.write_text("upstream\n", encoding="utf-8", newline="\n")
        self.patch_path = self.root / PATCH_PATH
        self.patch_path.parent.mkdir(parents=True)
        self.patch_path.write_text(TEST_PATCH, encoding="utf-8", newline="\n")
        self.namespace = load_patch_functions(self.root)
        self.apply = self.namespace["_apply_proton_patch"]

    def test_off_keeps_sources_unchanged(self):
        self.apply(False)
        self.assertEqual(self.original.read_text(), "upstream\n")
        self.assertFalse(self.added.exists())

    def test_repeated_on_off_builds_without_git_metadata(self):
        self.assertFalse((self.root / ".git").exists())
        for _ in range(2):
            self.apply(True)
            self.apply(True)
            self.assertEqual(self.original.read_text(), "ascend\n")
            self.assertEqual(self.added.read_text(), "new backend\n")
            self.apply(False)
            self.apply(False)
            self.assertEqual(self.original.read_text(), "upstream\n")
            self.assertFalse(self.added.exists())

    def test_apply_conflict_preserves_edits(self):
        self.original.write_text("user edit\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "no files were changed"):
            self.apply(True)
        self.assertEqual(self.original.read_text(), "user edit\n")
        self.assertFalse(self.added.exists())

    def test_restore_conflict_preserves_edits(self):
        self.apply(True)
        self.added.write_text("user edit\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.apply(False)
        self.assertEqual(self.original.read_text(), "ascend\n")
        self.assertEqual(self.added.read_text(), "user edit\n")

    def test_existing_added_file_is_not_overwritten(self):
        self.added.write_text("user file\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.apply(True)
        self.assertEqual(self.added.read_text(), "user file\n")
        self.assertEqual(self.original.read_text(), "upstream\n")

    def test_incomplete_patch_is_rejected(self):
        self.original.write_text("ascend\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.apply(True)
        self.assertFalse(self.added.exists())

    def test_missing_patch_is_reported(self):
        self.patch_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Proton patch"):
            self.apply(True)
        self.assertEqual(self.original.read_text(), "upstream\n")

    def test_build_hook_runs_before_cmake(self):
        function = next(node for node in SETUP_AST.body
                        if isinstance(node, ast.FunctionDef) and node.name == "patch_module")
        build_class = next(node for node in function.body
                           if isinstance(node, ast.ClassDef) and node.name == "CMakeBuild")
        observed = []
        original = self.original

        class BaseBuild:

            def run(self):
                observed.append(original.read_text())

        self.namespace.update({
            "_OrigCMakeBuild":
            BaseBuild,
            "_apply_triton_ascend_patch":
            lambda: None,
            "_clean_hitest_env":
            lambda: None,
            "os":
            os,
            "mod":
            SimpleNamespace(check_env_flag=lambda name, default: os.getenv(name, default).upper() in
                            ["ON", "1", "YES", "TRUE", "Y"]),
        })
        exec(compile(ast.Module(body=[build_class], type_ignores=[]), str(SETUP_PATH), "exec"), self.namespace)
        build = self.namespace["CMakeBuild"]()
        for flag in ("OFF", "ON", "1", "OFF"):
            with patch.dict(os.environ, TRITON_BUILD_PROTON=flag, TRITON_ENABLE_COVERAGE_HITEST="0"):
                build.run()
        self.assertEqual(observed, ["upstream\n", "ascend\n", "ascend\n", "upstream\n"])

    def test_real_patch_roundtrip(self):
        root = self.root / "real"
        source = root / "third_party/proton"
        shutil.copytree(ROOT / "third_party/proton", source)
        (root / PATCH_PATH).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / PATCH_PATH, root / PATCH_PATH)
        apply = load_patch_functions(root)["_apply_proton_patch"]
        apply(False)

        # Git for Windows may change checkout line endings during application.
        def snapshot():
            return {
                path.relative_to(source): path.read_bytes().replace(b"\r\n", b"\n")
                for path in source.rglob("*")
                if path.is_file()
            }

        before = snapshot()
        apply(True)
        self.assertTrue((source / "csrc/lib/Driver/GPU/CannApi.cpp").exists())
        apply(False)
        self.assertEqual(before, snapshot())


if __name__ == "__main__":
    unittest.main()
