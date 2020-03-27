if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

if(NOT DEFINED FLATBUFFERS_PATH)
  set(FLATBUFFERS_PATH $ENV{FLATBUFFERS_PATH})
endif()

include_directories(${FLATBUFFERS_PATH}/include)

add_llvm_library(MLIRTPU
  IR/TPUCompressUtil.cpp
  IR/TPUCompressUtilTest.cpp
  IR/TPUDialect.cpp
  IR/WeightFileOp.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  IR/TPUTensorSupport.cpp
  Analysis/TpuOpPrint.cpp
  Analysis/TpuOpStats.cpp
  Transforms/CompressWeight.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/ConvertScale.cpp
  Transforms/ConvertNormalize.cpp
  Transforms/FuseRelu.cpp
  Transforms/RefactorEltAndConv.cpp
  Transforms/MakeConvIcToEven.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Transforms/AssignLayerId.cpp
  Transforms/SplitCpuOp.cpp
  Transforms/ConvertPriorBoxToLoadWeight.cpp
  Transforms/ConvertCpuOp.cpp
  Transforms/ConvertConv2D.cpp
  Transforms/LowerToTG.cpp
  Transforms/group_ops/GroupOps.cpp
  Transforms/group_ops/NetGraph.cpp
  Transforms/group_ops/ImLayer.cpp
  Transforms/group_ops/Tensor.cpp
  Transforms/group_ops/GroupOptimizer.cpp
  Transforms/group_ops/Group.cpp
  Transforms/group_ops/LayerStage.cpp
  Transforms/group_ops/Steps.cpp
  Transforms/group_ops/LMemManager.cpp
  Transforms/group_ops/GMemManager.cpp
  Transforms/group_ops/MixNet.cpp
  Interpreter/TpuInterpreter.cpp
  Interpreter/NativeCpuImplementation.cpp
  Interpreter/CpuLayer_DetectionOutput.cpp
  Quantization/QuantizationArithmetic.cpp
  Quantization/ImportCalibrationTable.cpp
  Quantization/Quantization.cpp
  Quantization/QuantizeInt8.cpp
  Quantization/QuantizeBf16.cpp
  Quantization/GenLutInt8Table.cpp
  Quantization/GenLutBf16Table.cpp
  Quantization/GenTanHTable.cpp
  Quantization/GenSqrtTable.cpp
  Quantization/GenReciprocalTable.cpp
  Quantization/GenSigmoidTable.cpp
  Optimization/FuseLeakyRelu.cpp
  Backend/TpuTgCodegen.cpp
  Backend/TpuTlCodegen_Simple.cpp
  Backend/TpuTlCodegen_LG.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TPU
)

if (${USE_GPU})

  target_compile_definitions(MLIRTPU PUBLIC -DUSE_GPU)
  # CUDA PACKAGE
  find_package(CUDA REQUIRED)
  find_path(CUDNN_INCLUDE cudnn.h
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE} /usr/local/cuda/include /usr/include/
    DOC "Path to cuDNN include directory." )
  find_library(CUDNN_LIBRARY NAMES libcudnn.so
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
    DOC "Path to cuDNN library.")

  CUDA_COMPILE(CU_O Interpreter/GPUImplementation.cu OPTIONS -DUSE_GPU=1)
  cuda_add_library(MLIRGPUOps STATIC
    ${CU_O}
  )
  target_compile_definitions(MLIRGPUOps PUBLIC -DUSE_GPU)
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS MLIRGPUOps)
  target_link_libraries(MLIRGPUOps
    ${tpu_LINKER_LIBS}
    ${GPU_LINKER_LIBS}
    ${CUDNN_LIBRARY}
  )

  install(TARGETS MLIRGPUOps
    ${export_to_llvmexports}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT MLIRGPUOps
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT MLIRGPUOps
    RUNTIME DESTINATION bin COMPONENT MLIRGPUOps)

  target_link_libraries(MLIRTPU
    MLIRIR
    MLIRStandardOps
    LLVMSupport
    backend
    MLIRGPUOps
    ${tpu_LINKER_LIBS}
    ${GPU_LINKER_LIBS}
    ${CUDNN_LIBRARY}
  )
else()
  target_link_libraries(MLIRTPU
    MLIRIR
    MLIRStandardOps
    LLVMSupport
    backend
    ${tpu_LINKER_LIBS}
  )
endif()

add_dependencies(MLIRTPU
  MLIRTPUOpsIncGen
  MLIRTPUInterfaceIncGen
  MLIRTPUAttributeIncGen
  MLIRIR
  LLVMSupport
  CViBackend
)
add_subdirectory(Translate)
add_subdirectory(DeepFusion)
add_subdirectory(MemRefLowering)
