add_llvm_library(MLIRTPUMemRefLowering
  ConvertFuncToMemRef.cpp
  ConvertTgOpToMemRef.cpp
  ConvertTgOpToTensor.cpp
  AssignNeuronAddressMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
  )

target_link_libraries(MLIRTPUMemRefLowering
  MLIRIR
  MLIRTPU
  MLIRSupport
  backend)

add_dependencies(MLIRTPUTranslate
  CViBackend)
