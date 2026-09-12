// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT
#include "launcher_args.h"
#include <cstdlib>
#include <iostream>

// Header-only probe: compare native grid lowering with the independent Python
// program_grid reference without requiring a valid device binary or launch.
int main(int argc, char **argv) {
  if (argc != 6)
    return 2;
  TritonNpuLaunchSpecV1 spec{};
  spec.version = 1;
  spec.struct_size = sizeof(spec);
  spec.flags = std::strtoull(argv[1], nullptr, 10);
  spec.physical_blocks = std::strtoul(argv[2], nullptr, 10);
  spec.participant_factor = spec.coalesce_factor = spec.task_type = 1;
  spec.coalesce_axis = -1;
  int32_t input[]{std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5])};
  triton::ascend::validateSpec(spec);
  auto grid = triton::ascend::prepareGrid(spec, input);
  TritonNpuArgTypeV1 types[]{{TRITON_NPU_POINTER, -1}, {TRITON_NPU_I8, -1}};
  triton::ascend::ArgLayout layout(spec, types, 2);
  for (auto dim : grid.grid)
    std::cout << dim << ' ';
  std::cout << grid.logicalBlocks << ' ' << grid.physicalBlocks << ' '
            << layout.args[1].offset << ' ' << layout.originalGrid << ' '
            << layout.grid << ' ' << layout.debug << ' ' << layout.total
            << '\n';
}
