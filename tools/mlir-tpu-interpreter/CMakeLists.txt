set(LIBS
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRTPU
  MLIRStandardOps
  MLIRSupport
)
add_llvm_tool(mlir-tpu-interpreter
  mlir-tpu-interpreter.cpp
)
llvm_update_compile_flags(mlir-tpu-interpreter)
whole_archive_link(mlir-tpu-interpreter ${LIBS})
target_link_libraries(mlir-tpu-interpreter PRIVATE MLIRIR ${LIBS} LLVMSupport)
