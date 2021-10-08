cmake_minimum_required(VERSION 3.10)
project(custom_op_function CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(NOT DEFINED MLIR_INCLUDE)
  message(FATAL_ERROR "Please set MLIR_INCLUDE to point to the include path of mlir")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${MLIR_INCLUDE})
include_directories(${CNPY_INCLUDE})

add_library(CustomOpPlugin SHARED
            my_add/MyAddOp.cpp
            my_mul/MyMulOp.cpp)

install(TARGETS CustomOpPlugin DESTINATION .)
