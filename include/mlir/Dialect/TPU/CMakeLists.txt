# set(LLVM_TARGET_DEFINITIONS TPUOps.td)
# mlir_tablegen(TPUOps.h.inc -gen-op-decls)
# mlir_tablegen(TPUOps.cpp.inc -gen-op-defs)
# add_public_tablegen_target(MLIRTPUOpsIncGen)

add_mlir_dialect(TPUOps)

