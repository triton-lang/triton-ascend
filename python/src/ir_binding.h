#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <vector>

#include "python/src/ir.h"

namespace py = pybind11;

namespace ir {
extern py::class_<TritonOpBuilder> *getBuilderClass();
} // namespace ir
