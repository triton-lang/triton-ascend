add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  PluginUtils.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)

# triton-ascend 3.5 lacks triton/Version.h (upstream 3.7 generates it via
# CMake configure_file). Define TRITON_VERSION here so PluginUtils.cpp's
# plugin version checks compile.
target_compile_definitions(TritonTools PRIVATE TRITON_VERSION="3.5.0")
