from pathlib import Path
import typer
import re
import subprocess
import os
import shutil

def main(ut_path_str: str):
    base_path = Path(__file__).resolve().parent.parent
    triton_opt_path = base_path / "python/build/cmake.linux-x86_64-cpython-3.11/bin"
    os.environ["PATH"] = str(triton_opt_path) + ":" + os.environ["PATH"]

    ut_path = Path(ut_path_str)
    ut_fn = ut_path.stem
    tmp_dir = Path("/user_home/projects/triton-ascend/build/tmp")

    if tmp_dir.exists():
      shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    before = tmp_dir / (ut_fn + "_before_pr.mlir.txt")
    after_standardize = tmp_dir / (ut_fn + "_after_pr_after_standardize.mlir.txt")
    after_orig = tmp_dir / (ut_fn + "_after_pr_after_standardize_新的结果.mlir.txt")

    with ut_path.open() as f:
      first_line = next(f)

    args: list[str] = re.findall(r'\s+(-{1,2}[a-zA-Z0-9][a-zA-Z0-9_-]*)', first_line)

    shutil.copy(ut_path, before)

    subprocess.run([
      "triton-opt",
      "--ssbuf-standardize-op",
      ut_path,
      "-o",
      after_standardize
    ])

    subprocess.run([
      "triton-opt",
      after_standardize,
      "-o",
      after_orig
    ] + args)

if __name__ == "__main__":
  typer.run(main)
