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

# ============ Optional Ascend NPU support ============
# Opt-in: enabled via -DPROTON_ENABLE_NPU=ON or auto-detected from a CANN
# toolkit in the environment; silently disabled when CANN is absent.
option(PROTON_ENABLE_NPU "Enable Ascend NPU support in proton" OFF)

if(NOT ASCEND_INCLUDE_DIR)
  set(PROTON_CANN_HOME "")
  if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(PROTON_CANN_HOME "$ENV{ASCEND_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
    set(PROTON_CANN_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
  endif()
  if(PROTON_CANN_HOME)
    foreach(_cann_include_candidate
            "${PROTON_CANN_HOME}/include"
            "${PROTON_CANN_HOME}/runtime/include"
            "${PROTON_CANN_HOME}/aarch64-linux/include"
            "${PROTON_CANN_HOME}/x86_64-linux/include")
      if(EXISTS "${_cann_include_candidate}/acl/acl.h")
        set(ASCEND_INCLUDE_DIR "${_cann_include_candidate}")
        break()
      endif()
    endforeach()
  endif()
endif()

if(NOT PROTON_ENABLE_NPU AND ASCEND_INCLUDE_DIR AND EXISTS "${ASCEND_INCLUDE_DIR}/acl/acl.h")
  set(PROTON_ENABLE_NPU ON)
endif()

if(PROTON_ENABLE_NPU)
  if(NOT ASCEND_INCLUDE_DIR OR NOT EXISTS "${ASCEND_INCLUDE_DIR}/acl/acl.h")
    message(WARNING "PROTON_ENABLE_NPU requested but acl/acl.h not found; NPU support disabled")
    set(PROTON_ENABLE_NPU OFF)
  else()
    get_filename_component(PROTON_CANN_BASE_DIR "${ASCEND_INCLUDE_DIR}" DIRECTORY)
    find_library(ASCEND_CL_LIBRARY ascendcl
      PATHS "${PROTON_CANN_BASE_DIR}/lib64"
            "${PROTON_CANN_BASE_DIR}/lib"
            "${PROTON_CANN_BASE_DIR}/../lib64"
            "${PROTON_CANN_BASE_DIR}/../lib"
      NO_DEFAULT_PATH)
    if(NOT ASCEND_CL_LIBRARY)
      message(WARNING "libascendcl not found under CANN; NPU support disabled")
      set(PROTON_ENABLE_NPU OFF)
    else()
      message(STATUS "Ascend NPU support enabled (include: ${ASCEND_INCLUDE_DIR}, lib: ${ASCEND_CL_LIBRARY})")
    endif()
  endif()
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

  if(PROTON_ENABLE_NPU)
    target_include_directories(${name}
      SYSTEM PRIVATE
        "${ASCEND_INCLUDE_DIR}"
    )
    target_compile_definitions(${name} PRIVATE PROTON_ENABLE_NPU)
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

if(PROTON_ENABLE_NPU)
  target_link_libraries(proton PRIVATE ${ASCEND_CL_LIBRARY})
endif()
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
