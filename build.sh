source .venv/bin/activate

export LLVM_INSTALL_PREFIX=/Users/sky_miner/Documents/Project/huawei/llvm-install-old

cd python
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install

ln -sf /Users/sky_miner/Documents/Project/huawei/github-triton-ascend/python/build/cmake.macosx-11.0-arm64-cpython-3.12/compile_commands.json ../compile_commands.json
