#include "python/src/ir_binding.h"

namespace ir {

// Cast helpers: cast runs here in libtriton.so (same module as the
// module_local Value/Type/OpState pybind registrations) so type info is
// visible. Plugin .so files call these to avoid cross-module cast failures.
mlir::Value pyobj_to_value(py::object obj) {
  return py::cast<mlir::Value>(obj);
}
mlir::Type pyobj_to_type(py::object obj) { return py::cast<mlir::Type>(obj); }
py::object value_to_pyobj(mlir::Value v) { return py::cast(v); }
py::object type_to_pyobj(mlir::Type t) { return py::cast(t); }
py::object opstate_to_pyobj(mlir::OpState op) { return py::cast(op); }
std::vector<mlir::Value> pyobj_to_vecval(py::object obj) {
  return py::cast<std::vector<mlir::Value>>(obj);
}
std::vector<mlir::Type> pyobj_to_vectype(py::object obj) {
  return py::cast<std::vector<mlir::Type>>(obj);
}

} // namespace ir
