add_llvm_library(MLIRTPUDeepFusion
  SimpleDeepFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
  )

target_link_libraries(MLIRTPUDeepFusion
  MLIRIR
  MLIRTPU
  MLIRSupport
  backend)

add_dependencies(MLIRTPUTranslate
  CViBackend)
