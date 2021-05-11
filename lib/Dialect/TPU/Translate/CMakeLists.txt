if(NOT translate_LINKER_LIBS)
  set(translate_LINKER_LIBS "")
endif(NOT translate_LINKER_LIBS)

include_directories(lz4)

add_llvm_library(MLIRTPUTranslate
  MlirToBackendTranslate.cpp
  MlirToCviModelTranslate.cpp
  lz4/lz4_all.c
  lz4/xxhash.c
)

list(APPEND translate_LINKER_LIBS ssl crypto)
target_link_libraries(MLIRTPUTranslate
  MLIRTPU
  MLIRIR
  LLVMSupport
  MLIRSupport
  cvikernel
  ${translate_LINKER_LIBS})
