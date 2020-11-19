add_llvm_library(TPUCSupport
  TensorFile.cpp
  )
target_link_libraries(TPUCSupport MLIRSupport cnpy)
