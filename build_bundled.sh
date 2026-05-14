CMC_VER=20260511021009
NPUIR_URL="https://cmc-szv.clouddragon.huawei.com/cmcversion/index/componentVersionView?deltaId=14444239124824960&isSelect=Software&url_data=${CMC_VER}"

# 临时文件夹，用于测试合一包构建，之后改成从 cmc 自动获取
NPUIR_BIN_PATH=build/bishengir/bin
TARGET_BIN_PATH=third_party/ascend/backend/bishengir/bin
mkdir -p ${TARGET_BIN_PATH}

set -e

cp ${NPUIR_BIN_PATH}/* ${TARGET_BIN_PATH}

export TRITON_DISABLE_LINE_INFO=0
export TRITON_BUILD_WITH_CLANG_LLD=false
export LLVM_SYSPATH=/home/c00961524/llvm-install/llvm-fad32722-ubuntu-x64/
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$HOME/personal/oldroot.cmake -DTRITON_BUILD_UT=OFF -DLLVM_ENABLE_ASSERTIONS=ON"
export TRITON_WHEEL_NAME="triton-ascend"
export LD_LIBRARY_PATH=$LLVM_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH

uv run setup.py bdist_wheel
