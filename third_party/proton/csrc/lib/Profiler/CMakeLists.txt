set(PROTON_PROFILER_SOURCES
  Instrumentation/InstrumentationProfiler.cpp
  Instrumentation/Metadata.cpp
)

if(PROTON_BUILD_CUDA)
  list(APPEND PROTON_PROFILER_SOURCES
    Cupti/CuptiPCSampling.cpp
    Cupti/CuptiProfiler.cpp
    Instrumentation/CudaRuntime.cpp
  )
endif()

if(PROTON_BUILD_HIP)
  list(APPEND PROTON_PROFILER_SOURCES
    RocTracer/RoctracerProfiler.cpp
    Instrumentation/HipRuntime.cpp
  )
endif()

if(PROTON_BUILD_MSPTI)
  list(APPEND PROTON_PROFILER_SOURCES Mspti/MsptiProfiler.cpp)
endif()

add_proton_library(ProtonProfiler ${PROTON_PROFILER_SOURCES})
