# set(LLVM_TARGET_DEFINITIONS TPUOps.td)
# mlir_tablegen(TPUOps.h.inc -gen-op-decls)
# mlir_tablegen(TPUOps.cpp.inc -gen-op-defs)
# add_public_tablegen_target(MLIRTPUOpsIncGen)

add_mlir_dialect(TPUOps)

set(LLVM_TARGET_DEFINITIONS TPUInterface.td)
mlir_tablegen(TPUInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TPUInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTPUInterfaceIncGen)
