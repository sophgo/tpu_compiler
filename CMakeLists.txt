# MLIR project.
set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include ) # --src-root
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# To setup RPATH for external libraries, e.g. Caffe
# set(CMAKE_SKIP_BUILD_RPATH  FALSE)  # no need
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(add_mlir_dialect dialect)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  add_public_tablegen_target(MLIR${dialect}IncGen)

  # Generate Dialect Documentation
  tablegen(MLIR ${dialect}.md -gen-op-doc "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/Dialects/${dialect}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md)
  add_custom_target(${dialect}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(mlir-doc ${dialect}DocGen)
endfunction()

add_custom_target(mlir-doc)

# TODO: This is to handle the current static registration, but should be
# factored out a bit.
function(whole_archive_link target)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib ")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
    ENDFOREACH(LIB)
  elseif(MSVC)
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${LIB} ")
    ENDFOREACH(LIB)
  else()
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

# Build the CUDA conversions and run according tests if the NVPTX backend
# is available
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_CUDA_CONVERSIONS_ENABLED 1)
else()
  set(MLIR_CUDA_CONVERSIONS_ENABLED 0)
endif()

set(MLIR_CUDA_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir CUDA runner")

if(NOT DEFINED MKLDNN_PATH)
  set(MKLDNN_PATH $ENV{MKLDNN_PATH})
endif()
if(NOT DEFINED MKLDNN_PATH)
  message(FATAL_ERROR "Please set MKLDNN_PATH to point to the MKLDNN installation path")
else(NOT DEFINED MKLDNN_PATH)
  message(STATUS "MKLDNN_PATH ${MKLDNN_PATH}")
endif()
#add_definitions(-DUSE_MKLDNN=1)
include_directories(${MKLDNN_PATH}/include)
link_directories(${MKLDNN_PATH}/lib)

if(NOT DEFINED CVIKERNEL_PATH)
  set(MKLDNN_PATH $ENV{CVIKERNEL_PATH})
endif()
if(NOT DEFINED CVIKERNEL_PATH)
  message(FATAL_ERROR "Please set CVIKERNEL_PATH to point to the CVIKERNEL installation path")
else(NOT DEFINED CVIKERNEL_PATH)
  message(STATUS "CVIKERNEL_PATH ${CVIKERNEL_PATH}")
endif()
include_directories(${CVIKERNEL_PATH}/include)
link_directories(${CVIKERNEL_PATH}/lib)

include_directories( "include")
include_directories( ${MLIR_INCLUDE_DIR})

include_directories( "third_party/cnpy" )
include_directories( "externals/backend/include" )

add_subdirectory(include/mlir)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(test)
add_subdirectory(third_party)
add_subdirectory(externals)
add_subdirectory(bindings/python)
add_subdirectory(bindings/python/pybind)

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/mlir include/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${MLIR_INCLUDE_DIR}/mlir ${MLIR_INCLUDE_DIR}/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-mlir-headers
                             DEPENDS mlir-headers
                             COMPONENT mlir-headers)
  endif()
endif()
