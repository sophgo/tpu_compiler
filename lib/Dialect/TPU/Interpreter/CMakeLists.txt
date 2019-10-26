if(NOT interpreter_LINKER_LIBS)
  set(interpreter_LINKER_LIBS "")
endif(NOT interpreter_LINKER_LIBS)

add_llvm_library(MLIRTPUInterpreter
  TpuInterpreter.cpp
  NativeCpuImplementation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
  )

target_link_libraries(MLIRTPUInterpreter
  MLIRIR
  MLIRTPU
  MLIRSupport
  backend
  ${interpreter_LINKER_LIBS})

add_dependencies(MLIRTPUInterpreter
  CViBackend)
