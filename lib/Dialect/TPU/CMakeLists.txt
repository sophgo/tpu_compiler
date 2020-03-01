if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

add_llvm_library(MLIRTPU
  IR/TPUDialect.cpp
  IR/WeightFileOp.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  IR/TPUTensorSupport.cpp
  Analysis/TpuOpPrint.cpp
  Analysis/TpuOpStats.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/ConvertScale.cpp
  Transforms/ConvertNormalize.cpp
  Transforms/FuseRelu.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Transforms/AssignLayerId.cpp
  Transforms/ConvertPriorBoxToLoadWeight.cpp
  Transforms/LowerToTG.cpp
  Interpreter/TpuInterpreter.cpp
  Interpreter/NativeCpuImplementation.cpp
  Interpreter/CpuLayer_DetectionOutput.cpp
  Quantization/QuantizationArithmetic.cpp
  Quantization/ImportCalibrationTable.cpp
  Quantization/QuantizeInt8.cpp
  Quantization/QuantizeBf16.cpp
  Quantization/GenLutInt8Table.cpp
  Quantization/GenLutBf16Table.cpp
  Quantization/GenTanHTable.cpp
  Quantization/GenSqrtTable.cpp
  Quantization/GenReciprocalTable.cpp
  Quantization/GenSigmoidTable.cpp
  Optimization/FuseLeakyRelu.cpp
  Backend/TpuTgCodegen.cpp
  Backend/TpuTlCodegen_Simple.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)

add_dependencies(MLIRTPU
  MLIRTPUOpsIncGen
  MLIRTPUInterfaceIncGen
  MLIRTPUAttributeIncGen
  MLIRIR
  LLVMSupport
  CViBackend)
target_link_libraries(MLIRTPU
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  backend
  ${tpu_LINKER_LIBS})

add_subdirectory(Translate)
add_subdirectory(DeepFusion)
add_subdirectory(MemRefLowering)
