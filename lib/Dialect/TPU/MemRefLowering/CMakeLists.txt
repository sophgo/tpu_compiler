add_llvm_library(MLIRTPUMemRefLowering
  ConvertFuncToMemRef.cpp
  ConvertFuncToTensor.cpp
  ConvertTgOpToMemRef.cpp
  ConvertTgOpToTensor.cpp
  AssignNeuronAddressMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/tpuc
  )

target_link_libraries(MLIRTPUMemRefLowering
  MLIRIR
  MLIRTPU
  MLIRSupport
  cvikernel)
