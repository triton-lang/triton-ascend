import subprocess


def checkout_file(files):
    try:
        subprocess.run(["git", "checkout", "--"] + files, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"init code failed,list:{files}") from e


def init_triton_ascend_code():
    patch_files = [
        "CMakeLists.txt",
        "include/triton/Dialect/Triton/IR/TritonAttrDefs.td",
        "lib/Dialect/Triton/IR/Traits.cpp",
        "python/src/ir.cc",
        "python/triton/_utils.py",
        "python/triton/compiler/code_generator.py",
        "python/triton/compiler/compiler.py",
        "python/triton/compiler/errors.py",
        "python/triton/language/math.py",
        "python/triton/language/core.py",
        "python/triton/language/semantic.py",
        "python/triton/language/standard.py",
        "python/triton/runtime/interpreter.py",
        "python/triton/runtime/jit.py",
        "bin/RegisterTritonDialects.h",
        "bin/triton-opt.cpp",
        "bin/CMakeLists.txt",
    ]
    dev_patch_files = [
        "python/triton/runtime/autotuner.py",
    ]
    checkout_file(dev_patch_files)
    checkout_file(patch_files)
    print("init triton ascend successfully")


init_triton_ascend_code()
