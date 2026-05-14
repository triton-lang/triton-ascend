export LLVM_INSTALL_PREFIX=/home/wzw/workspace/build-env/llvm-20
export PYTHON=/home/wzw/workspace/build-env/venv311/bin/python
export MAX_JOBS=10

rm -rf python/dist

cd python && mkdir dist
rm -rf build
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
DEBUG=1 \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
${PYTHON} setup.py sdist bdist_wheel
