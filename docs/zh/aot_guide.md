# Triton-Ascend AOT编译使用指南

## 简介

**AOT（Ahead-of-Time，提前编译）** 把Triton kernel的编译从“首次调用时”提前到“构建期”：在构建环境中，`compile.py`按固化的特化参数（constexpr参数、dtype与对齐hint）把`@triton.jit`装饰的kernel编译成自包含C启动器的设备二进制（npubin）；`link.py`把同一kernel的多个特化版本链接成单一入口的分发器；生成的C代码与用户程序一起用gcc构建后，即可在运行环境由纯C/C++进程加载启动，执行结果与JIT（Just-In-Time，即时编译）一致。

AOT适合以下场景：

- **运行环境没有Python与编译器工具链**：只能部署纯C/C++程序，或需要把kernel集成进已有的C/C++服务；
- **消除首次调用延迟**：JIT在首次调用时现场编译，AOT把编译前移到构建期，提前编译好后在线服务启动即可执行；
- **形状与调优参数固定的推理kernel**：shape、dtype、tile配置在构建期确定，一次构建、多机分发。

AOT不用于运行期调优：特化参数在构建期固化，运行期不会重新编译；输入shape或对齐不满足任何已编译特化时，分发器会返回错误。需要按输入动态调优、或输入形状频繁变化的场景，请使用JIT方式。

## 使用前准备

- **构建环境**：与JIT相同的依赖（Python、CANN、torch_npu、triton-ascend），另需gcc（以及可选的g++）用于编译生成的C代码或C++调用程序；工具以模块方式提供：`python -m triton.tools.compile`与`python -m triton.tools.link`；
- **运行环境**：昇腾驱动与CANN运行库，以及launcher运行时依赖的动态库（torch_npu/torch的库目录与libpython共享库）；不需要Python解释器与编译器工具链。

一次编译的产物包括：

| 产物 | 说明 |
|---|---|
| `matmul_fp16.<hash>_<suffix>.c/.h` | 每个特化版本一套C stub，内嵌npubin字节数组 |
| `launcher_export.so` | 通用launcher运行时（启动策略随各kernel的C stub分发），同目录多个kernel共用，纯C进程经dlopen调用 |
| `launcher_cann.h` | NPU共享实现头（二进制注册与启动封装），编译C程序时需要 |
| `kernel.h / kernel.c` | link.py生成的单入口分发器 |

## 使用步骤

本文以固定形状的fp16矩阵乘为例，从编写kernel开始，到用`torch.matmul`完成精度比对结束：

1. 构建环境完成“编写kernel、编译特化、链接分发器、构建调用程序”四步；
2. 运行环境执行kernel并导出结果，再由Python脚本在具备torch的环境（通常是构建环境）做精度比对。示例代码用C编写，C++的调用方式见4.2末尾。

### 1. 编写kernel

与JIT完全一致，constexpr参数（BLOCK等）即特化维度：

```python
# kernel.py
import triton
import triton.language as tl

@triton.jit
def kernel(C, A, B, M, N, K,
           stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

### 2. 编译各特化版本

每个固化参数组合执行一条compile.py命令，`--signature`描述参数类型与特化方式：

```bash
python -m triton.tools.compile kernel.py -n kernel --signature "*fp32, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 16, 16, 16" -o matmul_fp16 --out-name matmul_fp16 -g "M/16, N/16, 1" -w 1
```

命令在kernel.py同级目录下生成`matmul_fp16.<hash>_<suffix>.c/.h`、`launcher_export.so`与`launcher_cann.h`。

signature的写法如下：

| 写法 | 含义 |
|---|---|
| `*fp32` / `*fp16` | 指针参数，映射为`void*` |
| `i32` / `i64` | 运行期标量参数 |
| `i32:16` | 16字节对齐特化：编译期以`tt.divisibility 16`特化，分发器按`% 16 == 0`匹配 |
| `i32:1` | 特化为常量1：不作为kernel入参，分发器按`== 1`匹配 |
| `16`（纯数字） | constexpr（如BLOCK），固化到二进制中 |

`-g`的网格表达式支持C算术（如`(n + 1023) / 1024`），表达式只能引用kernel的运行期标量参数名，便于处理非整除的输入。

> 注：constexpr值会拼入生成的C标识符，只支持整数；浮点常量（如`2.0`）会产生非法标识符导致编译失败。

### 3. 链接为单入口分发器

把各特化版本的`.h`一并传给link.py，生成`kernel.h/kernel.c`：

```bash
python -m triton.tools.link matmul_fp16.*.h -o kernel --backend npu
```

链接后运行期只有一个入口`matmul_fp16_default(stream, args...)`：分发器按实参的hint条件（对齐/输入值）匹配对应的特化版本，全部不匹配时返回错误。link.py同时生成`matmul_fp16(stream, args, algo_id)`入口与`algo_id`函数表。

### 4. 编写并构建C/C++调用程序

下面的`main.c`是完整的调用示例（C++调用见4.2末尾）：生成fp16输入、启动AOT kernel，并把输入与结果导出为`A.bin`/`B.bin`/`C.bin`。`main.c`和`torch.matmul`的精度比对由4.3的Python脚本完成。

#### 4.1 编写main.c

```c
// main.c
#include <acl/acl.h>
#include <stdint.h>
#include <stdio.h>

#include "kernel.h"

#define M 32
#define N 32
#define K 32

static uint16_t fp16_from_int(int v) {
    uint16_t sign = v < 0 ? 0x8000u : 0u;
    unsigned a = (unsigned)(v < 0 ? -v : v);
    if (a == 0) return sign;
    int e = 31 - __builtin_clz(a);
    unsigned mant = ((a << 10) >> e) & 0x3FFu;
    return (uint16_t)(sign | ((uint16_t)(e + 15) << 10) | (uint16_t)mant);
}

static int write_file(const char *path, const void *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        perror(path);
        return -1;
    }
    if (fwrite(data, 1, size, fp) != size) {
        perror(path);
        fclose(fp);
        return -1;
    }
    return fclose(fp) == 0 ? 0 : -1;
}

int main(void) {
    static uint16_t hA[M * K], hB[K * N];
    static float hC[M * N];

    /* Input: small integers exactly representable in fp16 */
    for (int i = 0; i < M * K; ++i) hA[i] = fp16_from_int((i * 7 + 3) % 17 - 8);
    for (int i = 0; i < K * N; ++i) hB[i] = fp16_from_int((i * 5 + 1) % 13 - 6);

    /* Device memory and stream */
    aclrtStream stream = NULL;
    void *dA = NULL, *dB = NULL, *dC = NULL;
    aclInit(NULL);
    aclrtSetDevice(0);
    aclrtMalloc(&dA, sizeof(hA), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dB, sizeof(hB), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC, sizeof(hC), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtCreateStream(&stream);

    /* Host -> device, launch the AOT kernel, device -> host */
    aclrtMemcpyAsync(dA, sizeof(hA), hA, sizeof(hA), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dB, sizeof(hB), hB, sizeof(hB), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtSynchronizeStream(stream);

    load_matmul_fp16(); /* register the embedded npubin */
    NPUresult ret = matmul_fp16_default(stream, dC, dA, dB, M, N, K,
                                        N, 1, K, 1, N, 1);
    if (ret != ASCEND_SUCCESS) {
        fprintf(stderr, "kernel launch failed: %d\n", ret);
        return 1;
    }
    aclrtSynchronizeStream(stream);
    aclrtMemcpyAsync(hC, sizeof(hC), dC, sizeof(hC), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    /* Export inputs and results */
    if (write_file("A.bin", hA, sizeof(hA)) != 0 || write_file("B.bin", hB, sizeof(hB)) != 0 ||
        write_file("C.bin", hC, sizeof(hC)) != 0) {
        return 1;
    }
    printf("matmul_fp16 M=%d N=%d K=%d: wrote A.bin (%dx%d fp16), B.bin (%dx%d fp16), C.bin (%dx%d fp32)\n",
           M, N, K, M, K, K, N, M, N);

    unload_matmul_fp16();
    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    aclrtDestroyStream(stream);
    aclFinalize();
    return 0;
}
```

#### 4.2 生成编译/链接参数并构建

C程序需要CANN、torch_npu与Python头文件/库的路径。为避免硬编码，下面这段脚本在构建环境中查询后端构建`npu_utils`时使用的同一套参数，并写成`aot_flags.sh`导入这些参数（只需要在构建环境执行一次，运行环境不需要执行该脚本）：

```bash
python - <<'PY' > aot_flags.sh
import os
import sys
import sysconfig
import warnings

warnings.filterwarnings("ignore")
import torch_npu
from torch.utils.cpp_extension import library_paths

from triton.backends.ascend import utils

ascend_path = str(utils._get_ascend_path())
has_runtime_header = os.path.exists(os.path.join(ascend_path, "include/experiment/runtime/runtime/rt.h"))
flags = list(
    utils._npu_ext_build_options("npu_utils", "torch", ascend_path, has_runtime_header, utils.backend_policy))
inc = [f for f in flags if f.startswith(("-I", "-D", "-w"))]
link = [f for f in flags if f.startswith(("-L", "-l"))]
lib_dirs = [f[2:] for f in link if f.startswith("-L")]
lib_dirs.append(os.path.join(os.path.dirname(torch_npu.__file__), "lib"))
lib_dirs += [d for d in library_paths() if d not in lib_dirs]
py_libdir = sysconfig.get_config_var("LIBDIR")
py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
print('INC="%s"' % " ".join(inc))
print('LINK="%s"' % " ".join(link + [f"-L{d}" for d in lib_dirs] + [f"-L{py_libdir}"] +
                      [f"-Wl,-rpath,{d}" for d in lib_dirs] + [f"-Wl,-rpath,{py_libdir}"]))
print('PYLIB="-Wl,--no-as-needed -l%s -Wl,--as-needed"' % py_lib)
PY
source aot_flags.sh

gcc main.c matmul_fp16.*.c kernel.c -I. $INC $LINK $PYLIB -Wl,-rpath,'$ORIGIN' -o demo
```

说明：

- 生成的`kernel.c`与各特化stub直接编入可执行文件，不需要额外的共享库；
- `-lpython`使用`--no-as-needed`保留在DT_NEEDED中：launcher运行时是Python扩展模块，进程里需要有Python数据符号；
- `-Wl,-rpath,'$ORIGIN'`让可执行文件在与`launcher_export.so`同目录时能找到共享库。

**用C++调用。** 生成的接口都是C ABI的普通函数，`kernel.h`里只有纯C声明；AOT运行时（`launcher_export.so`、`npu_utils.so`）也是C ABI的共享库，因此C++程序可以直接调用。有两种构建方式：

- **混合编译**（生成代码交给gcc）：C++源文件包含`kernel.h`时用`extern "C"`包起来，避免C++名字修饰：

  ```cpp
  extern "C" {
  #include "kernel.h"
  }
  ```

  ```bash
  gcc -c matmul_fp16.*.c kernel.c -I. $INC
  g++ main.cpp *.o -I. $INC $LINK $PYLIB -Wl,-rpath,'$ORIGIN' -o demo_cpp
  ```

- **全部由g++编译**（生成的C代码本身兼容C++）：直接`#include "kernel.h"`即可：

  ```bash
  g++ main.cpp matmul_fp16.*.c kernel.c -I. $INC $LINK $PYLIB -Wl,-rpath,'$ORIGIN' -o demo_cpp
  ```

  上面的`main.c`用g++直接编译同样可用（声明与定义都按C++编译，不需要`extern "C"`）。

两种方式得到的调用签名、运行方式与运行结果都与上面的C程序一致。

#### 4.3 精度对比脚本

`main.c`导出三个按行主序连续存放的二进制文件：`A.bin`（M×K，fp16）、`B.bin`（K×N，fp16）、`C.bin`（M×N，fp32）。下面的脚本读取这三个文件，用`torch.matmul`以float32累加计算作为参考结果，与kernel输出的全部M×N个元素逐一比较（不是抽样）：

```python
# compare_precision.py
import sys

import numpy as np
import torch

M, N, K = 32, 32, 32

a = torch.from_numpy(np.fromfile("A.bin", dtype=np.float16).reshape(M, K))
b = torch.from_numpy(np.fromfile("B.bin", dtype=np.float16).reshape(K, N))
out = torch.from_numpy(np.fromfile("C.bin", dtype=np.float32).reshape(M, N))

# Reference result
ref = torch.matmul(a.float(), b.float())

# Compare all M*N elements one by one
diff = (out - ref).abs()
max_abs = diff.max().item()
row, col = divmod(int(diff.argmax().item()), N)
scale = ref.abs().max().item()
max_rel = max_abs / scale if scale > 0 else max_abs

atol = 1e-3
mismatches = int((diff > atol).sum().item())
print(f"compared {M}x{N} = {out.numel()} elements")
print(f"max_abs_err={max_abs:.6e} at out[{row},{col}]  max_rel_err={max_rel:.6e}  "
      f"mismatches(>{atol:g})={mismatches}/{out.numel()}")
print(f"matmul_fp16 M={M} N={N} K={K}: {'PASS' if mismatches == 0 else 'FAIL'}")
sys.exit(0 if mismatches == 0 else 1)
```

脚本需要Python与torch，在具备该环境的位置执行（通常是构建环境；独立运行环境可以把三个`.bin`文件拷回构建环境再比对）。

#### 4.4 运行并比对精度

```bash
# Load CANN environment variables (replace with the actual installation path)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Add the library search paths required by the launcher runtime
# (libtorch_npu, libtorch, libpython).
# The python -c command below resolves them from the installed packages, so it
# requires Python and torch: run it in the build environment (or another
# environment with torch installed). On a runtime-only machine without Python,
# substitute the actual paths resolved in the build environment, e.g.
# /path/to/torch_npu/lib:/path/to/torch/lib:/path/to/python/lib.
export LD_LIBRARY_PATH="$(python -c 'import os, sysconfig, torch, torch_npu; print(":".join([os.path.join(os.path.dirname(torch_npu.__file__), "lib"), os.path.join(os.path.dirname(torch.__file__), "lib"), sysconfig.get_config_var("LIBDIR")]))'):$LD_LIBRARY_PATH"

./demo
python compare_precision.py
```

预期输出如下：

```text
matmul_fp16 M=32 N=32 K=32: wrote A.bin (32x32 fp16), B.bin (32x32 fp16), C.bin (32x32 fp32)
compared 32x32 = 1024 elements
max_abs_err=0.000000e+00 at out[0,0]  max_rel_err=0.000000e+00  mismatches(>0.001)=0/1024
matmul_fp16 M=32 N=32 K=32: PASS
```

#### 4.5 部署到运行环境

运行环境只需要可执行文件与运行时文件，不需要Python解释器、gcc或生成的C源码：

| 文件 | 说明 |
|---|---|
| `demo` | 已链接AOT产物的可执行文件 |
| `launcher_export.so` | 通用launcher运行时，与可执行文件放在同一目录（经`$ORIGIN`找到） |
| `npu_utils.so` | launcher运行时依赖的辅助库，构建时生成在triton缓存中 |

`npu_utils.so`的路径可以在构建环境中查询：

```bash
python -c "from triton.backends.ascend.driver import NPUUtils; print(NPUUtils().get_so_path())"
```

把它按`<TRITON_CACHE_DIR>/<hash>/npu_utils.so`的布局拷贝到运行环境，并设置`TRITON_CACHE_DIR`指向缓存根目录（运行环境与构建环境相同时，缓存已存在，无需额外设置）。运行环境还需要设置好`LD_LIBRARY_PATH`：CANN运行库、torch_npu/torch的库目录与libpython共享库。

`./demo`运行后在当前目录写出`A.bin`/`B.bin`/`C.bin`；`compare_precision.py`属于验证工具（需要Python与torch），不随运行环境部署。

如果把多个kernel打包成共享库，部署时还需把`libkernel.so`与可执行文件放在同一目录。

#### 4.6 多个kernel打包到同一个库

一个`kernel.py`里可以有多个`@triton.jit` kernel。把它们编进同一个库（或同一个可执行文件）时，对每个kernel各执行一次compile即可（例如用一个for循环依次编译），关键是`--out-name`（以及`-o`）要按kernel取不同的值。例如在matmul之外再增加一个relu kernel（示例取n=1024）：

```python
@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)
```

```bash
# One compile per kernel; later compiles do not affect the launch behavior of already-compiled kernels
for spec in \
  "kernel|matmul_fp16|*fp32, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 16, 16, 16|M/16, N/16, 1" \
  "relu_kernel|relu_fp32|*fp32, *fp32, i32, 1024|n/1024, 1, 1"; do
  IFS='|' read -r name out signature grid <<< "$spec"
  python -m triton.tools.compile kernel.py -n "$name" --signature "$signature" \
      -o "$out" --out-name "$out" -g "$grid" -w 1
done
```

link阶段对每个kernel **分别**执行一次link，`-o`取不同的文件名：

```bash
python -m triton.tools.link matmul_fp16.*.h -o dispatch_matmul --backend npu
python -m triton.tools.link relu_fp32.*.h   -o dispatch_relu   --backend npu
```

然后把两个kernel的stub与各自的dispatcher一起编进同一个共享库（下面两条构建命令需要在已source过`aot_flags.sh`的同一个shell中执行，`$INC`/`$LINK`/`$PYLIB`不会导出到子进程）：

```bash
gcc -fPIC -c matmul_fp16.*.c relu_fp32.*.c dispatch_matmul.c dispatch_relu.c -I. $INC
gcc -shared -o libkernel.so matmul_fp16.*.o relu_fp32.*.o dispatch_matmul.o dispatch_relu.o \
    $LINK -Wl,-rpath,'$ORIGIN' -ldl
```

调用程序生成`$INC`/`$LINK`/`$PYLIB`后链接这个库；运行时`libkernel.so`与`launcher_export.so`都放在可执行文件同一目录：

```bash
gcc main.c -I. $INC $LINK $PYLIB -Wl,-rpath,'$ORIGIN' -L. -lkernel -o demo_multi
```

C侧对每个kernel调用各自的`load_*`后即可正常启动：

```c
#include "dispatch_matmul.h"
#include "dispatch_relu.h"

load_matmul_fp16();
load_relu_fp32();
matmul_fp16_default(stream, dC, dA, dB, M, N, K, N, 1, K, 1, N, 1);
relu_fp32_default(stream, dX, dY, n);
```

注意：

- `--out-name`决定导出符号（`<name>_default`、`load_<name>`等），不同kernel不能重名；
- 不要把多个不同kernel的头文件传给同一次link；
- 所有kernel共用同一份`launcher_export.so`，部署时与库/可执行文件放在同一目录即可。

## 约束

- 多kernel需分别链接：link.py的模型是“单kernel多特化”，同一文件里的多个不同kernel应逐个单独link（`-o dispatch_<name>`），再把各自的`dispatch_*.c`一起编入同一个程序（可执行文件或共享库）；
- 并行执行多个编译进程时，torch的privateuse1后端自动加载可能静默失败，建议编译进程先显式`import torch_npu`；
- 生成的C stub自包含（`launcher_cann.h`提供`NPUresult/NPUstream`与`ASCEND_*`定义）；`launcher_cann.h`只在编译C程序时需要，运行环境不需要提供；
- 构建环境与运行环境的配置建议保持一致（如CANN版本、驱动版本、环境变量等），若不一致可能出现未知问题。
