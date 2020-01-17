if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

add_llvm_library(MLIRTPU
  IR/TPUDialect.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  Analysis/TpuOpPrint.cpp
  Analysis/TpuOpStats.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/FoldScale.cpp
  Transforms/MergeScaleIntoConv.cpp
  Transforms/FuseRelu.cpp
  Transforms/FuseEltwise.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Transforms/AssignLayerId.cpp
  Transforms/ConvertPriorBoxToLoadWeight.cpp
  Quantization/QuantizationArithmetic.cpp
  Quantization/ImportCalibrationTable.cpp
  Quantization/QuantizeInt8.cpp
  Quantization/QuantizeBf16.cpp
  Quantization/GenTanHTable.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)

add_dependencies(MLIRTPU
  MLIRTPUOpsIncGen
  MLIRIR
  LLVMSupport)
target_link_libraries(MLIRTPU
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  ${tpu_LINKER_LIBS})

add_subdirectory(Translate)
add_subdirectory(Interpreter)
add_subdirectory(DeepFusion)
