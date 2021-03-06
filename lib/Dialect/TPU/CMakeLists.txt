if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn ssl crypto)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized -Wno-unused-result")

if(NOT DEFINED FLATBUFFERS_PATH)
  set(FLATBUFFERS_PATH $ENV{FLATBUFFERS_PATH})
endif()

include_directories(${FLATBUFFERS_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Backend/include)

file(GLOB interpreter_sources Interpreter/core/*.cpp)

add_llvm_library(MLIRTPU
  IR/TPUCompressUtil.cpp
  IR/TPUCompressUtilTest.cpp
  IR/TPUDialect.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  IR/TPUTensorSupport.cpp
  Analysis/GenFakeWeightNpz.cpp
  Analysis/TpuOpPrint.cpp
  Analysis/TpuOpStats.cpp
  Transforms/CompressWeight.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/ConvertScale.cpp
  Transforms/ConvertClip.cpp
  Transforms/ConvertMulConst.cpp
  Transforms/ConvertNormalize.cpp
  Transforms/ConvertSwish.cpp
  Transforms/ConvertPermute.cpp
  Transforms/FuseRelu.cpp
  Transforms/FusePad.cpp
  Transforms/FuseReshape.cpp
  Transforms/MakeConvIcToEven.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Transforms/AssignLayerId.cpp
  Transforms/DivideOpsToFunc.cpp
  Transforms/AssignChipName.cpp
  Transforms/DivideOpsToSubFunc.cpp
  Transforms/GmemAllocator.cpp
  Transforms/GmemAllocatorMethod.cpp
  Transforms/ConvertPriorBoxToLoadWeight.cpp
  Transforms/ReorderOp.cpp
  Transforms/ConvertInterp.cpp
  Transforms/ConvertSwapChannel.cpp
  Transforms/ConvertLrn.cpp
  Transforms/ConvertUpsample.cpp
  Transforms/ConvertMatMul.cpp
  Transforms/ConvertFullyConnected.cpp
  Transforms/Canonicalizer.cpp
  Transforms/LowerToTG.cpp
  Interpreter/MlirModuleInterpreter.cpp
  ${interpreter_sources}
  Interpreter/NativeCpuImplementation.cpp
  Interpreter/CpuLayer_DetectionOutput.cpp
  Quantization/QuantizationArithmetic.cpp
  Quantization/ImportCalibrationTable.cpp
  Quantization/Quantization.cpp
  Quantization/QuantizeInt8.cpp
  Quantization/QuantizeBf16.cpp
  Optimization/SplitPool.cpp
  Optimization/FuseLeakyRelu.cpp
  Optimization/TgOpTile.cpp
  Optimization/GraphOpt.cpp
  Optimization/TgOpDivide.cpp
  Optimization/DoEarlyStride.cpp
  Optimization/AddTpuPreprocess.cpp
  Optimization/MergeConvConvPool.cpp
  Optimization/MergePermute.cpp
  Optimization/SimpleAnalysis.cpp
  Optimization/LayerGroup/GroupOps.cpp
  Optimization/LayerGroup/NetGraph.cpp
  Optimization/LayerGroup/ImLayer.cpp
  Optimization/LayerGroup/Tensor.cpp
  Optimization/LayerGroup/GroupOptimizer.cpp
  Optimization/LayerGroup/Group.cpp
  Optimization/LayerGroup/LayerStage.cpp
  Optimization/LayerGroup/Steps.cpp
  Optimization/LayerGroup/LMemManager.cpp
  Optimization/LayerGroup/LMemAllocMethod.cpp
  Optimization/LayerGroup/MixNet.cpp
  Optimization/LayerGroup/DeadCodeEliminate.cpp
  Optimization/LayerGroup/utils.cpp
  Optimization/LayerGroup/TdmaCycle.cpp
  Optimization/LayerGroup/TiuCycle.cpp
  Backend/TpuTgCodegen.cpp
  Backend/TpuTlCodegen_LG.cpp
  Plugin/CustomOpPlugin.cpp
  Backend/MachineInfo.cpp
  Backend/Kernel/CviBackendContext.cpp
  Backend/Kernel/TgBroadcastKernel.cpp
  Backend/Kernel/TgEltwiseKernel.cpp
  Backend/Kernel/TgMulConstKernel.cpp
  Backend/Kernel/TgFcKernel.cpp
  Backend/Kernel/TgFixedPoolingKernel.cpp
  Backend/Kernel/TgQuantKernel.cpp
  Backend/Kernel/TgDequantKernel.cpp
  Backend/Kernel/TgArgMaxKernel.cpp
  Backend/Kernel/TgConcatKernel.cpp
  Backend/Kernel/TgConvKernel.cpp
  Backend/Kernel/TgCropKernel.cpp
  Backend/Kernel/TgFixedDilateKernel.cpp
  Backend/Kernel/TgFixedLrnKernel.cpp
  Backend/Kernel/TgFillConstKernel.cpp
  Backend/Kernel/TgLutKernel.cpp
  Backend/Kernel/TgPadKernel.cpp
  Backend/Kernel/TgPermuteKernel.cpp
  Backend/Kernel/TgPixelShuffleKernel.cpp
  Backend/Kernel/TgPoolMaskKernel.cpp
  Backend/Kernel/TgReluKernel.cpp
  Backend/Kernel/TgReorgKernel.cpp
  Backend/Kernel/TgReverseKernel.cpp
  Backend/Kernel/TgReflectionPad.cpp
  Backend/Kernel/TgScaleKernel.cpp
  Backend/Kernel/TgScaleLutKernel.cpp
  Backend/Kernel/TgSwapChannelKernel.cpp
  Backend/Kernel/TgYuv420Kernel.cpp
  Backend/Kernel/TgCopyKernel.cpp
  Backend/Kernel/TgTileKernel.cpp
  Backend/Kernel/TgUpsampleKernel.cpp
  Backend/Kernel/TgZeroMaskKernel.cpp
  Backend/Kernel/TgReduceKernel.cpp
  Backend/Kernel/TgBf16Conv3dKernel.cpp
  Backend/Kernel/TgBf16GruKernel.cpp
  Backend/Kernel/TgBf16LayerNormKernel.cpp
  Backend/Kernel/TgBf16ConvFcKernel.cpp
  Backend/Kernel/TgBf16LstmKernel.cpp
  Backend/Kernel/TgBf16PoolingKernel.cpp
  Backend/Kernel/TgBf16Pooling3dKernel.cpp
  Backend/Kernel/TgBf16SoftmaxKernel.cpp
  Backend/Kernel/TgBf16StdKernel.cpp
  Backend/Kernel/TgBf16LrnKernel.cpp
  Backend/Kernel/TgBf16MatchTemplateKernel.cpp
  Backend/Kernel/TlConcat.cpp
  Backend/Kernel/TlConv.cpp
  Backend/Kernel/TlCrop.cpp
  Backend/Kernel/TlDeconv.cpp
  Backend/Kernel/TlEltwise.cpp
  Backend/Kernel/TlLayerNorm.cpp
  Backend/Kernel/TlLeakyRelu.cpp
  Backend/Kernel/TlLrn.cpp
  Backend/Kernel/TlLut.cpp
  Backend/Kernel/TlMulConst.cpp
  Backend/Kernel/TlPad.cpp
  Backend/Kernel/TlPooling.cpp
  Backend/Kernel/TlPrelu.cpp
  Backend/Kernel/TlRelu.cpp
  Backend/Kernel/TlQuant.cpp
  Backend/Kernel/TlScale.cpp
  Backend/Kernel/TlScaleLut.cpp
  Backend/Kernel/TlSwapChannel.cpp
  Backend/Kernel/TlTdma.cpp
  Backend/Kernel/TlUpsample.cpp
  Backend/Kernel/TlPixelShuffle.cpp
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
    LLVMSupport
    cvikernel
    MLIRGPUOps
    ${tpu_LINKER_LIBS}
    ${GPU_LINKER_LIBS}
    ${CUDNN_LIBRARY}
  )
else()
  target_link_libraries(MLIRTPU
    MLIRIR
    LLVMSupport
    cvikernel
    ${tpu_LINKER_LIBS}
  )
endif()

add_dependencies(MLIRTPU
  MLIRTPUOpsIncGen
  MLIRTPUInterfaceIncGen
  MLIRTPUAttributeIncGen
  MLIRTPUPassIncGen
  MLIRIR
  LLVMSupport
)
add_subdirectory(Translate)
