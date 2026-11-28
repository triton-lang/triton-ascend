# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import glob
import os
import pytest
import re
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

import triton

kernel_utils_src = """
import triton

@triton.jit
def mul(x, y):
    return x * y
"""

kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)):
      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
      # We accumulate along the K dimension.
      accumulator += tl.dot(a, b)
      # Advance the ptrs to the next K block.
      a_ptrs += BLOCK_K * stride_ak
      b_ptrs += BLOCK_K * stride_bk

  c = kernel_utils.mul(accumulator, accumulator)
  # Write back the block of the output matrix C with masks.
  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
"""

test_utils_src = """
#include <acl/acl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def is_npu():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "npu"


def _torch_npu_library_dir():
    import torch_npu

    return os.path.join(os.path.dirname(torch_npu.__file__), "lib")


def _npu_toolchain():
    """Reuse the ascend backend's own NPU extension compile/link flags.

    These are the same flags used to build npu_utils and the launcher
    runtime, so the AOT C harness always matches the installed toolchain
    (CANN include/lib layout, torch_npu paths and ABI defines) without
    hard-coding any path.
    """
    from triton.backends.ascend import utils as ascend_utils

    asc_path = str(ascend_utils._get_ascend_path())
    has_runtime_header = os.path.exists(os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h"))
    flags = list(
        ascend_utils._npu_ext_build_options("npu_utils", "torch", asc_path, has_runtime_header,
                                            ascend_utils.backend_policy))
    include_flags = [flag for flag in flags if flag.startswith(("-I", "-D", "-w"))]
    link_flags = [flag for flag in flags if flag.startswith(("-L", "-l"))]
    lib_dirs = [flag[2:] for flag in link_flags if flag.startswith("-L")]
    # The launcher chain (npu_utils/launcher runtime .so) links torch
    # extension libraries (libtorch_python.so etc.); the plain C test
    # process must be able to load them, so extend the search path with
    # torch's library dirs.
    from torch.utils.cpp_extension import library_paths

    lib_dirs += [lib_dir for lib_dir in [*library_paths(), _torch_npu_library_dir()] if lib_dir not in lib_dirs]
    python_libdir = sysconfig.get_config_var("LIBDIR")
    python_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ld_library_path = [*lib_dirs] + ([python_libdir] if python_libdir else [])
    return include_flags, link_flags, lib_dirs, python_libdir, python_lib, ld_library_path


def gen_kernel_library_npu(dir, libname):
    include_flags, link_flags, lib_dirs, _, _, _ = _npu_toolchain()
    c_files = glob.glob(os.path.join(dir, "*.c"))
    # Compile each generated launcher source, then link the shared library
    # against the CANN/torch_npu runtime libs its rt*/aclrt* calls need
    # (they are referenced, so --as-needed keeps them).
    subprocess.run(
        ["gcc"] + c_files + include_flags + ["-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))
    command = ["gcc", *o_files, "-shared", "-o", libname]
    for lib_dir in lib_dirs:
        command += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
    command += link_flags + ["-ldl"]
    subprocess.run(command, check=True, cwd=dir)


def gen_test_bin_npu(dir, M, N, K, exe="test", algo_id=0):
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize ACL handles
  aclrtStream stream = NULL;
  void *A = NULL, *B = NULL, *C = NULL;
  aclError err = 0;
  aclInit(NULL);
  int32_t device_id = 0;
  aclrtSetDevice(device_id);
  aclrtMalloc(&A, M * K * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&B, K * N * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&C, M * N * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtCreateStream(&stream);
  load_matmul_fp16();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  memset(hA, 0, M*K*2);
  memset(hB, 0, K*N*2);
  read_csv_to_buffer(argv[1], hA, M*K);
  read_csv_to_buffer(argv[2], hB, K*N);
  aclrtMemcpyAsync(A, M*K*2, hA, M*K*2, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtMemcpyAsync(B, K*N*2, hB, K*N*2, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);

  // launch kernel
  int ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = matmul_fp16(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);
  aclrtSynchronizeStream(stream);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  aclrtMemcpyAsync(hC, M*N*4, C, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free acl handles
  unload_matmul_fp16();
  aclrtFree(A);
  aclrtFree(B);
  aclrtFree(C);
  aclrtDestroyStream(stream);
  aclFinalize();
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)

    include_flags, link_flags, lib_dirs, python_libdir, python_lib, _ = _npu_toolchain()
    command = ["gcc", "test.c"]
    command.extend(include_flags)
    # The backend's link flags already carry the right -L/-l order. Keep
    # libpython in DT_NEEDED even though test.c never references it
    # directly: --as-needed would otherwise drop it, and the launcher
    # artifacts need Python data symbols in the process when dlopened.
    command.extend(link_flags)
    if python_libdir:
        command += [f"-L{python_libdir}", f"-Wl,-rpath,{python_libdir}"]
    for lib_dir in lib_dirs:
        command += [f"-Wl,-rpath,{lib_dir}"]
    command += [
        "-Wl,--no-as-needed", f"-l{python_lib}", "-Wl,--as-needed", "-ldl", "-L", dir, "-l", "kernel", "-o", exe
    ]
    subprocess.run(command, check=True, cwd=dir)


def _npu_run_env(tmp_dir):
    _, _, _, _, _, ld_library_path = _npu_toolchain()
    env = os.environ.copy()
    paths = [tmp_dir] + ld_library_path + [env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(path for path in paths if path)
    return env


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch_npu, runpy, sys; sys.argv = [sys.argv[1]] + sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            kernel_path,
        ],
        env=_npu_run_env(dir),
        check=True,
        cwd=dir,
    )


# Edge case kernel with no specialization
def compile_aot_kernel_no_specialization(dir, kernel_path, dtype, BM, BN, BK):
    # compile all desired configs
    sig = f"*fp32, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}"
    name = f"matmul_{dtype}"
    grid = f"M/{BM}, N/{BN}, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="kernel",
        out_name=name,
        out_path=name,
        num_warps=1,
        grid=grid,
        kernel_path=kernel_path,
    )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    # compile all desired configs
    for ha, hb in ha_hb_hints:
        sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
        name = f"matmul_{dtype}"
        grid = f"M/{BM}, N/{BN}, 1"
        _compile_kernel(
            dir=dir,
            signature=sig,
            kernel_name="kernel",
            out_name=name,
            out_path=name,
            num_warps=1,
            grid=grid,
            kernel_path=kernel_path,
        )


def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    cmd = [sys.executable, linker_path] + h_files + ["-o", "kernel"] + ["--backend", "npu"]
    subprocess.run(cmd, check=True, cwd=dir)


def generate_matmul_test_data(dir, M, N, K):
    a = np.random.randn(M * K).astype(np.float16).reshape((M, K))
    b = np.random.randn(M * K).astype(np.float16).reshape((K, N))
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for x, path in [(a, a_path), (b, b_path)]:
        x.view(np.int16).ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path


def test_compile_link_matmul_no_specialization():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK)

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_compile_link_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":16", ":16")])
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_launcher_has_no_available_kernel():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":1", ":1")])

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        result = subprocess.run(
            ["./test", a_path, b_path, c_path],
            env=_npu_run_env(tmp_dir),
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        # It should fail since the launcher requires all the strides be 1 while they are not.
        assert result.returncode == -6
        assert "kernel launch failed" in result.stderr


# Several compile.py runs into one output directory

SIG_NOSPEC = ("*fp32, *fp16, *fp16, i32, i32, i32, "
              "i32, i32, i32, i32, i32, i32, 16, 16, 16")
SIG_ALIGNED = ("*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, "
               "i32:16, i32:1, i32:16, i32:1, i32:16, i32:1, 16, 16, 16")
SIG_SQUARE = "*fp16:16, *fp16:16, i32, i32, i32, 256"
GRID_SQUARE = "n/256, 1, 1"

square_kernel_src = """
@triton.jit
def square_fp16(Y, X, n, y_stride, x_stride, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(X + offs * x_stride, mask=mask)
    tl.store(Y + offs * y_stride, v * v, mask=mask)
"""

multi_kernel_src = kernel_src + square_kernel_src

two_kernel_host_src = test_utils_src.replace('#include "kernel.h"',
                                             '#include "matmul_dispatch.h"\n#include "square_dispatch.h"') + """
#include <stdlib.h>

static void write_i16_csv(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", (int)buffer[i]);
        if (i < size - 1) fprintf(file, ",");
    }
    fclose(file);
}

int main(int argc, char **argv) {
  int M = 16, N = 16, K = 16, n = 1024;
  aclrtStream stream = NULL;
  void *A = NULL, *B = NULL, *C = NULL, *X = NULL, *Y = NULL;
  aclInit(NULL);
  aclrtSetDevice(0);
  aclrtMalloc(&A, M * K * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&B, K * N * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&C, M * N * 4, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&X, n * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&Y, n * 2, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtCreateStream(&stream);
  load_matmul_fp16();
  load_square_fp16();
  int16_t *hA = (int16_t*)malloc(M * K * 2);
  int16_t *hB = (int16_t*)malloc(K * N * 2);
  int16_t *hX = (int16_t*)malloc(n * 2);
  memset(hA, 0, M * K * 2);
  memset(hB, 0, K * N * 2);
  memset(hX, 0, n * 2);
  read_csv_to_buffer(argv[1], hA, M * K);
  read_csv_to_buffer(argv[2], hB, K * N);
  read_csv_to_buffer(argv[3], hX, n);
  aclrtMemcpyAsync(A, M * K * 2, hA, M * K * 2, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtMemcpyAsync(B, K * N * 2, hB, K * N * 2, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtMemcpyAsync(X, n * 2, hX, n * 2, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  aclrtSynchronizeStream(stream);
  int ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  if (ret != 0) fprintf(stderr, "matmul launch failed\\n");
  assert(ret == 0);
  ret = square_fp16_default(stream, Y, X, n, 1, 1);
  if (ret != 0) fprintf(stderr, "square launch failed\\n");
  assert(ret == 0);
  aclrtSynchronizeStream(stream);
  int32_t *hC = (int32_t*)malloc(M * N * 4);
  int16_t *hY = (int16_t*)malloc(n * 2);
  memset(hC, 0, M * N * 4);
  memset(hY, 0, n * 2);
  aclrtMemcpyAsync(hC, M * N * 4, C, M * N * 4, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtMemcpyAsync(hY, n * 2, Y, n * 2, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  aclrtSynchronizeStream(stream);
  write_buffer_to_csv(argv[4], hC, M * N);
  write_i16_csv(argv[5], hY, n);
  unload_matmul_fp16();
  unload_square_fp16();
  aclrtFree(A); aclrtFree(B); aclrtFree(C); aclrtFree(X); aclrtFree(Y);
  aclrtDestroyStream(stream);
  aclFinalize();
  free(hA); free(hB); free(hX); free(hC); free(hY);
  return 0;
}
"""


def _compile_kernel_npu(out_dir, signature, kernel_path, kernel_name="kernel", out_name="matmul_fp16",
                        grid="M/16, N/16, 1", num_warps=1):
    before = set(glob.glob(os.path.join(out_dir, "*.h")))
    _compile_kernel(out_dir, signature, kernel_name, out_name, out_name, num_warps, grid, kernel_path)
    after = set(glob.glob(os.path.join(out_dir, "*.h")))
    new = sorted(after - before)
    return [os.path.basename(path) for path in (new or sorted(after))]


def link_aot_kernel_headers(dir, headers, out_name):
    """Link headers into one dispatcher; only the first original kernel gets <kernel>_default."""
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")
    subprocess.run([sys.executable, linker_path, *headers, "-o", out_name, "--backend", "npu"], check=True, cwd=dir)


def _gcc_exe(dir, src_name, exe):
    include_flags, link_flags, lib_dirs, python_libdir, python_lib, _ = _npu_toolchain()
    command = ["gcc", src_name]
    command.extend(include_flags)
    command.extend(link_flags)
    if python_libdir:
        command += [f"-L{python_libdir}", f"-Wl,-rpath,{python_libdir}"]
    for lib_dir in lib_dirs:
        command += [f"-Wl,-rpath,{lib_dir}"]
    command += [
        "-Wl,--no-as-needed", f"-l{python_lib}", "-Wl,--as-needed", "-ldl", "-L", dir, "-l", "kernel", "-o", exe
    ]
    subprocess.run(command, check=True, cwd=dir)


def test_aot_per_kernel_launcher_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = write_triton_kernels(tmp_dir, multi_kernel_src, kernel_utils_src)
        _compile_kernel_npu(tmp_dir, SIG_ALIGNED, kernel_path, out_name="matmul_fp16")
        _compile_kernel_npu(tmp_dir, SIG_SQUARE, kernel_path, kernel_name="square_fp16", out_name="square_fp16",
                            grid=GRID_SQUARE)
        c_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".c"))
        assert len(c_files) == 2, c_files
        spec_names = set()
        for name in c_files:
            text = open(os.path.join(tmp_dir, name), encoding="utf-8").read()
            match = re.search(r"static const npu_aot_launch_spec_t (\w+)_spec", text)
            assert match, name
            func_name = match.group(1)
            assert func_name.startswith(("matmul_fp16_", "square_fp16_")), (name, func_name)
            assert f"static const npu_aot_arg_type_t {func_name}_types[]" in text, name
            assert f"&{func_name}_spec" in text, name
            assert f"{func_name}_types" in text, name
            assert "npuAotLaunch(" in text, name
            spec_names.add(func_name)
        assert len(spec_names) == 2, spec_names
        so = os.path.join(tmp_dir, "launcher_export.so")
        nm = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True, check=True).stdout
        exported = {line.split()[-1] for line in nm.splitlines() if line.strip()}
        assert {"triton_launch_kernel_create", "triton_launch_kernel_run", "triton_launch_kernel"} <= exported, \
            sorted(exported)


def test_aot_two_kernels_independent():
    np.random.seed(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = write_triton_kernels(tmp_dir, multi_kernel_src, kernel_utils_src)
        matmul_headers = _compile_kernel_npu(tmp_dir, SIG_ALIGNED, kernel_path, out_name="matmul_fp16")
        square_headers = _compile_kernel_npu(tmp_dir, SIG_SQUARE, kernel_path, kernel_name="square_fp16",
                                             out_name="square_fp16", grid=GRID_SQUARE)
        matmul_headers = [header for header in matmul_headers if header != "launcher_cann.h"]
        square_headers = [header for header in square_headers if header != "launcher_cann.h"]
        # Keep matmul first so link.py emits matmul_fp16_default for the host.
        link_aot_kernel_headers(tmp_dir, matmul_headers + square_headers, "kernel")
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        gen_test_bin_npu(tmp_dir, 16, 16, 16)

        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, 16, 16, 16)
        subprocess.run(["./test", a_path, b_path, c_path], env=_npu_run_env(tmp_dir), check=True, cwd=tmp_dir)

        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((16, 16)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("order", ["matmul_first", "square_first"])
def test_aot_two_kernels_both_launch(order):
    np.random.seed(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = write_triton_kernels(tmp_dir, multi_kernel_src, kernel_utils_src)
        steps = [("matmul", SIG_ALIGNED, "kernel", "matmul_fp16", "M/16, N/16, 1"),
                 ("matmul", SIG_NOSPEC, "kernel", "matmul_fp16", "M/16, N/16, 1"),
                 ("square", SIG_SQUARE, "square_fp16", "square_fp16", GRID_SQUARE)]
        if order == "square_first":
            steps.reverse()
        matmul_headers, square_headers = [], []
        for kind, signature, kernel_name, out_name, grid in steps:
            headers = _compile_kernel_npu(tmp_dir, signature, kernel_path, kernel_name=kernel_name, out_name=out_name,
                                          grid=grid)
            headers = [header for header in headers if header != "launcher_cann.h"]
            (matmul_headers if kind == "matmul" else square_headers).extend(headers)
        assert len(matmul_headers) == 2, matmul_headers
        assert len(square_headers) == 1, square_headers
        link_aot_kernel_headers(tmp_dir, matmul_headers, "matmul_dispatch")
        link_aot_kernel_headers(tmp_dir, square_headers, "square_dispatch")
        gen_kernel_library_npu(tmp_dir, "libkernel.so")
        with open(os.path.join(tmp_dir, "test.c"), "w") as file:
            file.write(two_kernel_host_src)
        _gcc_exe(tmp_dir, "test.c", "test")

        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, 16, 16, 16)
        n = 1024
        x = (np.arange(n, dtype=np.int32) % 17 - 8).astype(np.float16)
        x_path = os.path.join(tmp_dir, "x.csv")
        y_path = os.path.join(tmp_dir, "y.csv")
        x.view(np.int16).tofile(x_path, sep=",")
        subprocess.run(["./test", a_path, b_path, x_path, c_path, y_path], env=_npu_run_env(tmp_dir), check=True,
                       cwd=tmp_dir)

        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((16, 16)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)
        y = np.genfromtxt(y_path, delimiter=",", dtype=np.int16).view(np.float16)
        np.testing.assert_allclose(y.astype(np.float32), x.astype(np.float32)**2, rtol=1e-3, atol=1e-3)
