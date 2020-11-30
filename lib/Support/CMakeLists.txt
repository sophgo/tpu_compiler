set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
add_llvm_library(TPUCSupport
  TensorFile.cpp
  )
target_link_libraries(TPUCSupport MLIRSupport cnpy)
