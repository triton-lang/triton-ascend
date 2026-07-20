add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  # PluginUtils.cpp is built as TritonPluginUtils STATIC (see below).

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)

# STATIC (not OBJECT) so only libtriton.so links it via TRITON_LIBRARIES;
# plugin .so files include only the header (declarations) and resolve
# symbols (TritonPlugin::load, loadPlugins, etc.) against libtriton.so
# at runtime -> single shared plugins list.
add_library(TritonPluginUtils STATIC PluginUtils.cpp)
target_link_libraries(TritonPluginUtils PUBLIC MLIRIR MLIRLLVMDialect)
if(TRITON_BUILD_PYTHON_MODULE)
  target_link_libraries(TritonPluginUtils PUBLIC pybind11::headers)
endif()
target_compile_definitions(TritonPluginUtils PRIVATE TRITON_VERSION="3.5.0")
