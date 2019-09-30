if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

add_llvm_library(MLIRTPU
  IR/TPUDialect.cpp
  IR/DialectRegistration.cpp
  Analysis/TpuOpStats-v0.cpp
  Analysis/TpuOpStats.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/FoldScale.cpp
  Transforms/FuseScaleIntoConv.cpp
  Interpreter/TpuInterpreter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)
add_dependencies(MLIRTPU MLIRTPUOpsIncGen MLIRIR LLVMSupport)
target_link_libraries(MLIRTPU MLIRIR MLIRStandardOps LLVMSupport ${tpu_LINKER_LIBS})

add_subdirectory(Translate)