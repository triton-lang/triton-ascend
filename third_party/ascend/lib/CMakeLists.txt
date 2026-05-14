add_subdirectory(AutoBlockify)
add_subdirectory(Dialect)
add_subdirectory(TritonToAnnotation)
add_subdirectory(TritonToHFusion)
add_subdirectory(TritonToHIVM)
add_subdirectory(TritonToGraph)
add_subdirectory(TritonToLinalg)
add_subdirectory(Utils)
add_subdirectory(DiscreteMaskAccessConversion)
add_subdirectory(TritonToUnstructure)
add_subdirectory(TritonToLLVM)
add_subdirectory(TritonToStructured)
add_subdirectory(DynamicCVPipeline)
# todo: this code will be removed in version 530.
add_subdirectory(TritonAffinityOpt)

if(TRITON_ENABLE_COVERAGE_HITEST)
  set(_instrument_targets
      DiscreteMaskAccessConversion
      TritonToAnnotation
      TritonToHFusion
      TritonToHIVM
      TritonToLinalg
      TritonToLLVM
      TritonToStructured
      TritonToUnstructure
      MLIRTritonNPUUtils           # from Utils
      TritonAscendIR               # from Dialect/TritonAscend/IR
      TritonStructuredIR           # from Dialect/TritonStructured/IR
      AutoBlockify
      TritonAffinityOpt
  )

  foreach(_target ${_instrument_targets})
      if(TARGET ${_target})
          set_target_properties(${_target} PROPERTIES
              RULE_LAUNCH_COMPILE "hitestwrapper"
              RULE_LAUNCH_LINK "hitestwrapper"
          )
          message(STATUS "Enabled hitestwrapper for target: ${_target}")
      else()
          message(WARNING "Target ${_target} not found, please check the actual target name")
      endif()
  endforeach()
endif()
