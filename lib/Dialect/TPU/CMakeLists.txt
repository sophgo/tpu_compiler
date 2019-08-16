add_llvm_library(MLIRTPU
  IR/TPUDialect.cpp
  IR/DialectRegistration.cpp
  Transforms/TpuOpStats-v0.cpp
  Transforms/TpuOpStats.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)
add_dependencies(MLIRTPU MLIRTPUOpsIncGen MLIRIR LLVMSupport)
target_link_libraries(MLIRTPU MLIRIR MLIRStandardOps LLVMSupport)

add_subdirectory(Translate)