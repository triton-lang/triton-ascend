project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Configure optional backends =============
# Community builds retain both GPU backends. Downstream builds opt out explicitly.
option(PROTON_BUILD_CUDA "Build the CUPTI and CUDA instrumentation backends" ON)
option(PROTON_BUILD_HIP "Build the ROCtracer and HIP instrumentation backends" ON)
option(PROTON_BUILD_MSPTI "Build the MSPTI backend" OFF)
option(PROTON_BUILD_CANN "Build the CANN device and instrumentation runtime" OFF)

function(proton_require_header directory header backend)
  if(NOT DEFINED ${directory} OR NOT EXISTS "${${directory}}/${header}")
    message(FATAL_ERROR "${backend} is enabled but ${header} is missing from ${directory}. Set ${directory} or explicitly disable ${backend}.")
  endif()
endfunction()

if(PROTON_BUILD_CUDA)
  proton_require_header(CUPTI_INCLUDE_DIR cupti.h PROTON_BUILD_CUDA)
  proton_require_header(CUPTI_INCLUDE_DIR cuda.h PROTON_BUILD_CUDA)
endif()
if(PROTON_BUILD_HIP)
  proton_require_header(ROCTRACER_INCLUDE_DIR roctracer/roctracer.h PROTON_BUILD_HIP)
  proton_require_header(ROCTRACER_INCLUDE_DIR hip/hip_runtime_api.h PROTON_BUILD_HIP)
endif()
if(PROTON_BUILD_MSPTI)
  if(NOT PROTON_BUILD_CANN)
    message(FATAL_ERROR "PROTON_BUILD_MSPTI requires PROTON_BUILD_CANN")
  endif()
  proton_require_header(MSPTI_INCLUDE_DIR mspti.h PROTON_BUILD_MSPTI)
endif()
if(PROTON_BUILD_CANN)
  proton_require_header(CANN_INCLUDE_DIR acl/acl.h PROTON_BUILD_CANN)
endif()
message(STATUS "Proton backends: CUDA=${PROTON_BUILD_CUDA}; HIP=${PROTON_BUILD_HIP}; MSPTI=${PROTON_BUILD_MSPTI}; CANN=${PROTON_BUILD_CANN}")

if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

if(PROTON_BUILD_CANN)
  get_filename_component(PROTON_CANN_BASE_DIR "${CANN_INCLUDE_DIR}" DIRECTORY)
  find_library(CANN_CL_LIBRARY ascendcl
    HINTS "${PROTON_CANN_BASE_DIR}/lib64"
          "${PROTON_CANN_BASE_DIR}/lib"
          "${PROTON_CANN_BASE_DIR}/../lib64"
          "${PROTON_CANN_BASE_DIR}/../lib"
    REQUIRED)
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

if(PROTON_BUILD_CANN)
  target_link_libraries(proton PRIVATE ${CANN_CL_LIBRARY})
endif()
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
