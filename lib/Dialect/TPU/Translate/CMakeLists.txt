if(NOT translate_LINKER_LIBS)
  set(translate_LINKER_LIBS "")
endif(NOT translate_LINKER_LIBS)

# for caffe importer
if(NOT DEFINED CAFFE_PATH)
  message(FATAL_ERROR "Please set CAFFE_PATH to point to the caffe source installation")
else(NOT DEFINED CAFFE_PATH)
  message(STATUS "CAFFE_PATH ${CAFFE_PATH}")
endif()
include(${CAFFE_PATH}/share/Caffe/CaffeConfig.cmake)
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CAFFE_PATH}/mkl/mklml/include)
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
list(APPEND translate_LINKER_LIBS
  ${Caffe_LIBRARIES})
# use followings if build caffe with make rather than cmake
if(0)
  include_directories(${CAFFE_PATH}/include)
  link_directories(${CAFFE_PATH}/lib)
  add_definitions(-DCPU_ONLY)
  list(APPEND translate_LINKER_LIBS
    caffe
    caffeproto
    protobuf
    boost_system)
endif()

add_llvm_library(MLIRTPUTranslate
  CaffeToMlirTranslate.cpp
  MlirToBackendTranslate.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
  )

target_link_libraries(MLIRTPUTranslate
  MLIRIR
  MLIRTPU
  MLIRSupport
  backend
  ${translate_LINKER_LIBS})

add_dependencies(MLIRTPUTranslate
  CViBackend)
