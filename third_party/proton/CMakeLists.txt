project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Detect optional backends =============
set(PROTON_BUILD_CUDA OFF)
if(DEFINED CUPTI_INCLUDE_DIR
   AND EXISTS "${CUPTI_INCLUDE_DIR}/cupti.h"
   AND EXISTS "${CUPTI_INCLUDE_DIR}/cuda.h")
  set(PROTON_BUILD_CUDA ON)
endif()

set(PROTON_BUILD_HIP OFF)
if(DEFINED ROCTRACER_INCLUDE_DIR
   AND EXISTS "${ROCTRACER_INCLUDE_DIR}/roctracer/roctracer.h"
   AND EXISTS "${ROCTRACER_INCLUDE_DIR}/hip/hip_runtime_api.h")
  set(PROTON_BUILD_HIP ON)
endif()

set(PROTON_BUILD_MSPTI OFF)
if(DEFINED MSPTI_INCLUDE_DIR AND EXISTS "${MSPTI_INCLUDE_DIR}/mspti.h")
  set(PROTON_BUILD_MSPTI ON)
endif()

set(PROTON_BUILD_CANN OFF)
if(DEFINED CANN_INCLUDE_DIR AND EXISTS "${CANN_INCLUDE_DIR}/acl/acl.h")
  set(PROTON_BUILD_CANN ON)
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

  target_include_directories(${name}
    PRIVATE
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  if(PROTON_BUILD_CUDA)
    target_include_directories(${name} PRIVATE "${CUPTI_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE PROTON_BUILD_CUDA)
  endif()
  if(PROTON_BUILD_HIP)
    # Use system to skip warnings caused by legacy clang compilers.
    target_include_directories(${name} SYSTEM PRIVATE "${ROCTRACER_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE PROTON_BUILD_HIP __HIP_PLATFORM_AMD__)
  endif()
  if(PROTON_BUILD_MSPTI)
    target_include_directories(${name} PRIVATE "${MSPTI_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE PROTON_BUILD_MSPTI)
    if(DEFINED MSPTI_LIB_DIR)
      target_compile_definitions(${name} PRIVATE MSPTI_LIB_DIR=${MSPTI_LIB_DIR})
    endif()
  endif()
  if(PROTON_BUILD_CANN)
    target_include_directories(${name} PRIVATE "${CANN_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE PROTON_BUILD_CANN)
  endif()
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
