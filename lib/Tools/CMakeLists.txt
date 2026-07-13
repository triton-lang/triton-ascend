add_triton_library(TritonTools
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
target_compile_definitions(TritonPluginUtils PRIVATE TRITON_VERSION="3.5.0")
