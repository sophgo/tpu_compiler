if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn bmkernel)

add_llvm_library(MLIRTPU
  IR/TPUDialect.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  Analysis/TpuOpStats-v0.cpp
  Analysis/TpuOpStats.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/FoldScale.cpp
  Transforms/FuseScaleIntoConv.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Quantization/QuantizationArithmetic.cpp
  Quantization/ImportCalibrationTable.cpp
  Quantization/QuantizeInt8.cpp
  Interpreter/TpuInterpreter.cpp
  Interpreter/NativeCpuImplementation.cpp
  Interpreter/BM1880v2BackendContext.cpp
  Interpreter/dma_legacy_bmkernel.cpp
  Interpreter/conv_parallel_bmkernel.cpp
  Interpreter/conv_parallel_bmkernel_qdm.cpp
  Interpreter/fc_parallel_bmkernel.cpp
  Interpreter/pooling_bmkernel.cpp
  Interpreter/relu_bmkernel.cpp
  Interpreter/eltwise_bmkernel.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)
add_dependencies(MLIRTPU MLIRTPUOpsIncGen MLIRIR LLVMSupport)
target_link_libraries(MLIRTPU MLIRIR MLIRStandardOps LLVMSupport ${tpu_LINKER_LIBS})

add_subdirectory(Translate)