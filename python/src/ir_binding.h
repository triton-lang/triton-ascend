#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <vector>

#include "python/src/ir.h"

namespace py = pybind11;

namespace ir {
extern py::class_<TritonOpBuilder> *getBuilderClass();

// Cast helpers: must run in libtriton.so (same module as the Value/Type/
// OpState pybind registrations, which are module_local). Plugin .so files
// use these to convert py::object <-> C++ types without cross-module cast
// failures (module_local type info is not visible across modules).
mlir::Value pyobj_to_value(py::object obj);
mlir::Type pyobj_to_type(py::object obj);
py::object value_to_pyobj(mlir::Value v);
py::object type_to_pyobj(mlir::Type t);
py::object opstate_to_pyobj(mlir::OpState op);
std::vector<mlir::Value> pyobj_to_vecval(py::object obj);
std::vector<mlir::Type> pyobj_to_vectype(py::object obj);
} // namespace ir
