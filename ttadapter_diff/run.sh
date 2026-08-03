#!/usr/bin/env bash

# 设置一旦报错就停止执行
set -e

# 颜色定义，方便看日志
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 参数校验
if [ "$#" -lt 2 ]; then
    echo -e "${RED}错误: 参数不足！${NC}"
    echo -e "用法: ./run_all.sh <基线文件夹> <对比文件夹> [输出csv路径]"
    echo -e "示例: ./run_all.sh ./compiles_res ./compiles_res_pr total_diff.csv"
    exit 1
fi

BASE_DIR=$1
CMP_DIR=$2
OUTPUT_CSV=${3:-"diff.csv"}

# ==============================================================================
# 动态获取当前脚本所在的绝对路径，确保后续调用 python 脚本时路径绝对正确
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}          Triton-Ascend 算子一键式 Diff 工具        ${NC}"
echo -e "${GREEN}==================================================${NC}"

# 1. 校验输入目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}错误: 基线目录不存在 -> $BASE_DIR${NC}"
    exit 1
fi

if [ ! -d "$CMP_DIR" ]; then
    echo -e "${RED}错误: 对比目录不存在 -> $CMP_DIR${NC}"
    exit 1
fi

# 去除路径末尾的斜杠，统一格式
BASE_DIR="${BASE_DIR%/}"
CMP_DIR="${CMP_DIR%/}"

BASE_HASHED_DIR="${BASE_DIR}_hashed"
CMP_HASHED_DIR="${CMP_DIR}_hashed"

# 2. 对基线文件夹生成 Hash
echo -e "\n${YELLOW}[步骤 1/3] 正在对基线文件夹生成 Hash...${NC}"
echo -e "输入: $BASE_DIR  -->  输出: $BASE_HASHED_DIR"
python3 "$SCRIPT_DIR/hash_pkl.py" "$BASE_DIR" "$BASE_HASHED_DIR"

# 3. 对对比文件夹生成 Hash
echo -e "\n${YELLOW}[步骤 2/3] 正在对对比文件夹生成 Hash...${NC}"
echo -e "输入: $CMP_DIR  -->  输出: $CMP_HASHED_DIR"
python3 "$SCRIPT_DIR/hash_pkl.py" "$CMP_DIR" "$CMP_HASHED_DIR"

# 4. 执行 Cmp 比对
echo -e "\n${YELLOW}[步骤 3/3] 正在比对两个 Hash 文件夹并生成报告...${NC}"
echo -e "基线: $BASE_HASHED_DIR"
echo -e "对比: $CMP_HASHED_DIR"
echo -e "CSV结果: $OUTPUT_CSV"

python3 "$SCRIPT_DIR/cmp.py" "$BASE_HASHED_DIR" "$CMP_HASHED_DIR" "$OUTPUT_CSV"

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}  一键式比对成功完成！请查看结果: $OUTPUT_CSV ${NC}"
echo -e "${GREEN}==================================================${NC}"