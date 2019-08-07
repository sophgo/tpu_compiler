add_llvm_library(MLIRTPUTranslate
  CaffeToMlirTranslate.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
  )

target_link_libraries(MLIRTPUTranslate
  MLIRIR
  MLIRTPU
  MLIRSupport)