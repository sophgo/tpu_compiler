cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pybind)

pybind11_add_module(pybind pybind.cpp)

whole_archive_link(mlir-tpu-interpreter
  MLIRStandardOps
  MLIRTransforms
  MLIRTranslation
)

target_link_libraries(pybind PRIVATE
  MLIRTransforms
  MLIRTranslation
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRAffineToStandard
  MLIRExecutionEngine
  MLIRTPU
  MLIRTPUInterpreter
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMCore
  LLVMSupport
  LLVMX86Desc
  LLVMX86Info
  LLVMX86CodeGen
)

