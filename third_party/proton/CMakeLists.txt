# =============================================================================
# Proton CMakeLists.txt - PURE CMAKE SOLUTION
# Fixed error: undefined symbol _ZN6proton4cuda9getDeviceEm
# · Added dummy implementations of cuda::getDevice and hip::getDevice (target_sources)
# · RoctracerProfiler dummy remains
# · include_directories(BEFORE) is placed first
# Works on pure Ascend (MSPTI=ON, CUPTI=OFF, ROCTRACER=OFF)
# =============================================================================

project(Proton CXX)

set(PROTON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(PROTON_EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)

file(GLOB_RECURSE PROTON_SRC
  ${PROTON_SRC_DIR}/lib/*.cpp
  ${PROTON_SRC_DIR}/${PROJECT_NAME}.cpp
)

option(PROTON_ENABLE_CUPTI    "Enable NVIDIA CUPTI + CUDA" OFF)
option(PROTON_ENABLE_ROCTRACER "Enable AMD ROCTRACER" OFF)
option(PROTON_ENABLE_MSPTI    "Enable Huawei Ascend MSPTI (mspti.so)" OFF)

message(STATUS "=== Proton build configuration ===")
message(STATUS "  PROTON_ENABLE_CUPTI     : ${PROTON_ENABLE_CUPTI}")
message(STATUS "  PROTON_ENABLE_ROCTRACER  : ${PROTON_ENABLE_ROCTRACER}")
message(STATUS "  PROTON_ENABLE_MSPTI     : ${PROTON_ENABLE_MSPTI}")
message(STATUS "===================================")

# -----------------------------------------------------------------------------
#STUB-MECHANISM (the very first)
# -----------------------------------------------------------------------------

set(PROTON_STUBS_DIR "${CMAKE_CURRENT_BINARY_DIR}/proton_stubs")
include_directories(BEFORE ${PROTON_STUBS_DIR})

file(MAKE_DIRECTORY "${PROTON_STUBS_DIR}/Driver/GPU")
file(MAKE_DIRECTORY "${PROTON_STUBS_DIR}/Profiler/Cupti")
file(MAKE_DIRECTORY "${PROTON_STUBS_DIR}/Profiler/RocTracer")

# NVIDIA stubs
if(NOT PROTON_ENABLE_CUPTI)
  file(WRITE "${PROTON_STUBS_DIR}/Driver/GPU/CudaApi.h" [=[
#ifndef PROTON_DRIVER_GPU_CUDA_H_
#define PROTON_DRIVER_GPU_CUDA_H_
#include "Driver/Device.h"
namespace proton { namespace cuda { Device getDevice(uint64_t index); } }
#endif
]=])
  file(WRITE "${PROTON_STUBS_DIR}/Driver/GPU/CuptiApi.h" [=[
#ifndef PROTON_DRIVER_GPU_CUPTI_H_
#define PROTON_DRIVER_GPU_CUPTI_H_
namespace proton { namespace cupti {} }
#endif
]=])
  file(WRITE "${PROTON_STUBS_DIR}/Profiler/Cupti/CuptiProfiler.h" [=[
#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_
#include "Profiler/Profiler.h"
namespace proton { class CuptiProfiler : public Profiler { public: static CuptiProfiler& instance() { static CuptiProfiler inst; return inst; } Profiler& enablePCSampling() { return *this; } virtual void doStart() override {} virtual void doFlush() override {} virtual void doStop() override {} virtual ~CuptiProfiler() = default; }; }
#endif
]=])
endif()

# AMD RoctracerProfiler stub
if(NOT PROTON_ENABLE_ROCTRACER)
  file(WRITE "${PROTON_STUBS_DIR}/Profiler/RocTracer/RoctracerProfiler.h" [=[
#ifndef PROTON_PROFILER_ROCTRACER_ROCTRACERPROFILER_H_
#define PROTON_PROFILER_ROCTRACER_ROCTRACERPROFILER_H_
#include "Profiler/Profiler.h"
namespace proton { class RoctracerProfiler : public Profiler { public: RoctracerProfiler(); virtual ~RoctracerProfiler(); static RoctracerProfiler& instance() { static RoctracerProfiler inst; return inst; } virtual void doStart() override {} virtual void doFlush() override {} virtual void doStop() override {}; }; }
#endif
]=])
endif()

# -----------------------------------------------------------------------------
#STRICT EXCLUSION of original vendor files
# -----------------------------------------------------------------------------
if(NOT PROTON_ENABLE_CUPTI)
  list(FILTER PROTON_SRC EXCLUDE REGEX "CuptiProfiler\\.cpp$|CuptiPCSampling\\.cpp$|CudaApi\\.cpp$|CuptiApi\\.cpp$")
endif()

if(NOT PROTON_ENABLE_ROCTRACER)
  list(FILTER PROTON_SRC EXCLUDE REGEX "RoctracerProfiler\\.cpp$|GPUProfiler\\.cpp$|HipApi\\.cpp$|HsaApi\\.cpp$|RoctracerApi\\.cpp$")
endif()

add_library(proton SHARED ${PROTON_SRC})

# -----------------------------------------------------------------------------
#DUMMY-IMPLEMENTS (resolve undefined symbols)
# -----------------------------------------------------------------------------
if(NOT PROTON_ENABLE_CUPTI)
  set(DUMMY_CUDA "${CMAKE_CURRENT_BINARY_DIR}/dummy_cuda_getDevice.cpp")
  file(WRITE ${DUMMY_CUDA} [=[
#include "Driver/GPU/CudaApi.h"
#include "Driver/Device.h"
namespace proton {
namespace cuda {
Device getDevice(uint64_t index) {
  return Device();   // dummy — backend отключён
}
} // namespace cuda
} // namespace proton
]=])
  target_sources(proton PRIVATE ${DUMMY_CUDA})
  message(STATUS "Proton: added dummy_cuda_getDevice.cpp (fixes _ZN6proton4cuda9getDeviceEm)")
endif()

if(NOT PROTON_ENABLE_ROCTRACER)
  set(DUMMY_ROC "${CMAKE_CURRENT_BINARY_DIR}/dummy_RoctracerProfiler.cpp")
  file(WRITE ${DUMMY_ROC} [=[
#include "Profiler/RocTracer/RoctracerProfiler.h"
namespace proton {
RoctracerProfiler::RoctracerProfiler() {}
RoctracerProfiler::~RoctracerProfiler() {}
}
]=])
  target_sources(proton PRIVATE ${DUMMY_ROC})
  message(STATUS "Proton: added dummy_RoctracerProfiler.cpp")
endif()

if(NOT PROTON_ENABLE_ROCTRACER)
  set(DUMMY_HIP "${CMAKE_CURRENT_BINARY_DIR}/dummy_hip_getDevice.cpp")
  file(WRITE ${DUMMY_HIP} [=[
#include "Driver/GPU/HipApi.h"
#include "Driver/Device.h"
namespace proton {
namespace hip {
Device getDevice(uint64_t index) {
  return Device();   // dummy — backend отключён
}
} // namespace hip
} // namespace proton
]=])
  target_sources(proton PRIVATE ${DUMMY_HIP})
  message(STATUS "Proton: added dummy_hip_getDevice.cpp")
endif()

# -----------------------------------------------------------------------------
#The rest of the stubs (AMD)
# -----------------------------------------------------------------------------
if(NOT PROTON_ENABLE_ROCTRACER)
  file(WRITE "${PROTON_STUBS_DIR}/Driver/GPU/HipApi.h" [=[
#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_
#include "Driver/Device.h"
namespace proton { namespace hip { Device getDevice(uint64_t index); } }
#endif
]=])
  file(WRITE "${PROTON_STUBS_DIR}/Driver/GPU/HsaApi.h" [=[
#ifndef PROTON_DRIVER_GPU_HSA_H_
#define PROTON_DRIVER_GPU_HSA_H_
namespace proton { namespace hsa {} }
#endif
]=])
  file(WRITE "${PROTON_STUBS_DIR}/Driver/GPU/RoctracerApi.h" [=[
#ifndef PROTON_DRIVER_GPU_ROCTRACER_H_
#define PROTON_DRIVER_GPU_ROCTRACER_H_
namespace proton { namespace roctracer {} }
#endif
]=])
  file(WRITE "${PROTON_STUBS_DIR}/Profiler/RocTracer/GPUProfiler.h" [=[
#ifndef PROTON_PROFILER_ROCTRACER_GPUPROFILER_H_
#define PROTON_PROFILER_ROCTRACER_GPUPROFILER_H_
namespace proton { namespace roctracer {} }
#endif
]=])
endif()

# Реальные include + define
if(PROTON_ENABLE_CUPTI)
  if(NOT CUDA_INCLUDE_DIR OR NOT CUPTI_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA_INCLUDE_DIR and CUPTI_INCLUDE_DIR required when PROTON_ENABLE_CUPTI=ON")
  endif()
  include_directories(${CUDA_INCLUDE_DIR})
  include_directories(${CUPTI_INCLUDE_DIR})
  if(DEFINED CUPTI_LIB_DIR)
    add_compile_definitions(CUPTI_LIB_DIR=${CUPTI_LIB_DIR})
  endif()
  target_compile_definitions(proton PRIVATE PROTON_ENABLE_CUPTI)
endif()

if(PROTON_ENABLE_ROCTRACER)
  if(NOT ROCTRACER_INCLUDE_DIR)
    message(FATAL_ERROR "ROCTRACER_INCLUDE_DIR required when PROTON_ENABLE_ROCTRACER=ON")
  endif()
  include_directories(SYSTEM ${ROCTRACER_INCLUDE_DIR})
  target_compile_definitions(proton PRIVATE PROTON_ENABLE_ROCTRACER __HIP_PLATFORM_AMD__)
endif()

if(PROTON_ENABLE_MSPTI)
  if(NOT MSPTI_INCLUDE_DIR)
    message(FATAL_ERROR "MSPTI_INCLUDE_DIR required when PROTON_ENABLE_MSPTI=ON")
  endif()
  include_directories(${MSPTI_INCLUDE_DIR})
  if(DEFINED MSPTI_LIB_DIR)
    add_compile_definitions(MSPTI_LIB_DIR=${MSPTI_LIB_DIR})
  endif()
  target_compile_definitions(proton PRIVATE PROTON_ENABLE_MSPTI)
endif()

# Required dependencies
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()
include_directories(${JSON_INCLUDE_DIR})
include_directories(${PROTON_SRC_DIR}/include)
include_directories(${PROTON_EXTERN_DIR})

# Python / pybind11
if(PYTHON_INCLUDE_DIRS)
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${PYBIND11_INCLUDE_DIR})
else()
  find_package(Python3 REQUIRED Interpreter Development)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
endif()

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

target_link_libraries(proton PRIVATE ${Python_LIBRARIES} ${PROTON_PYTHON_LDFLAGS})
=======
project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")

# ============ Add subdirectory with proton tests ============
add_subdirectory(test)

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

target_link_libraries(proton PRIVATE Python3::Module)
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
