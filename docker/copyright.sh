#!/bin/bash

triton_ascend_version="3.2.1"
cann_version="9.0.0"
torch_npu_version="2.7.1"
python_version="3.11"

GREEN='\033[1;32m'
NC='\033[0m'
echo "=========================================="
echo -e "${GREEN} Triton-Ascend Dev Container${NC}"
echo "=========================================="
echo "The versions mapping is as follows:"
echo "   Triton-Ascend : ${triton_ascend_version}"
echo "   CANN          : ${cann_version} "
echo "   Torch-npu     : ${torch_npu_version}"
echo "   Python        : ${python_version}"
echo "=========================================="
echo "Container image Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved."
echo "This container image and its contents are governed by the Huawei Container License Agreement (\"License\"). By pulling and using the container, you accept the terms and conditions of this License."
echo "A copy of this License is made available in this container at: https://www.hiascend.com/en/legal/ascendhub-download"
echo "Note: You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software."
