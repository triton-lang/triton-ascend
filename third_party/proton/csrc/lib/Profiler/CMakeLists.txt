add_proton_library(ProtonProfiler
	Cupti/CuptiPCSampling.cpp
	Cupti/CuptiProfiler.cpp
	RocTracer/RoctracerProfiler.cpp
	Instrumentation/InstrumentationProfiler.cpp
	Instrumentation/CudaRuntime.cpp
	Instrumentation/HipRuntime.cpp
	Instrumentation/Metadata.cpp
)

if(PROTON_ENABLE_NPU)
	target_sources(ProtonProfiler PRIVATE Instrumentation/AscendRuntime.cpp)
endif()
