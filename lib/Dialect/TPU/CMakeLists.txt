if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-pedantic -fno-strict-aliasing")

if(NOT DEFINED FLATBUFFERS_PATH)
  set(FLATBUFFERS_PATH $ENV{FLATBUFFERS_PATH})
endif()

include_directories(${FLATBUFFERS_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Backend/include)

add_llvm_library(MLIRTPU
  IR/TPUCompressUtil.cpp
  IR/TPUCompressUtilTest.cpp
  IR/TPUDialect.cpp
  IR/WeightFileOp.cpp
  IR/DialectRegistration.cpp
  IR/TPUOperationSupport.cpp
  IR/TPUTensorSupport.cpp
  Analysis/GenFakeWeightNpz.cpp
  Analysis/TpuOpPrint.cpp
  Analysis/TpuOpStats.cpp
  Transforms/CompressActivation.cpp
  Transforms/CompressWeight.cpp
  Transforms/ConvertBnToScale.cpp
  Transforms/ConvertScale.cpp
  Transforms/ConvertClip.cpp
  Transforms/ConvertNormalize.cpp
  Transforms/ConvertSwishToRelu.cpp
  Transforms/ConvertPermute.cpp
  Transforms/FuseRelu.cpp
  Transforms/FusePad.cpp
  Transforms/RefactorEltAndConv.cpp
  Transforms/MakeConvIcToEven.cpp
  Transforms/AssignWeightAddress.cpp
  Transforms/AssignNeuronAddress.cpp
  Transforms/AssignLayerId.cpp
  Transforms/DivideOpsToFunc.cpp
  Transforms/AssignChipName.cpp
  Transforms/DivideOpsToSubFunc.cpp
  Transforms/GmemAllocator.cpp
  Transforms/ConvertPriorBoxToLoadWeight.cpp
  Transforms/ReorderOp.cpp
  Transforms/ConvertInterp.cpp
  Transforms/ConvertConv2D.cpp
  Transforms/ConvertUpsample.cpp
  Transforms/ConvertTile.cpp
  Transforms/ConvertQuantOp.cpp
  Transforms/ConvertPower.cpp
  Transforms/ConvertMatMul.cpp
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
  Transforms/group_ops/MixNet.cpp
  Transforms/group_ops/DeadCodeEliminate.cpp
  Transforms/group_ops/utils.cpp
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
  Optimization/ConvTile.cpp
  Optimization/FuseLeakyRelu.cpp
  Optimization/FullyConnectedTile.cpp
  Optimization/TgOpTile.cpp
  Backend/TpuTgCodegen.cpp
  Backend/TpuTlCodegen_Simple.cpp
  Backend/TpuTlCodegen_LG.cpp
  Plugin/CustomOpPlugin.cpp
  Backend/MachineInfo.cpp
  Backend/Kernel/CviBackendContext.cpp
  Backend/Kernel/TgFixedEltwiseKernel.cpp
  Backend/Kernel/TgFixedFcKernel.cpp
  Backend/Kernel/TgFixedPoolingKernel.cpp
  Backend/Kernel/TgCastKernel.cpp
  Backend/Kernel/TgFixedConcatKernel.cpp
  Backend/Kernel/TgConvKernel.cpp
  Backend/Kernel/TgFixedCropKernel.cpp
  Backend/Kernel/TgFixedMacConstKernel.cpp
  Backend/Kernel/TgFixedDilateKernel.cpp
  Backend/Kernel/TgFixedLrnKernel.cpp
  Backend/Kernel/TgLutKernel.cpp
  Backend/Kernel/TgMixedPrecision.cpp
  Backend/Kernel/TgPadKernel.cpp
  Backend/Kernel/TgPermuteKernel.cpp
  Backend/Kernel/TgPixelShuffleKernel.cpp
  Backend/Kernel/TgFixedPowerKernel.cpp
  Backend/Kernel/TgReluKernel.cpp
  Backend/Kernel/TgReorgKernel.cpp
  Backend/Kernel/TgFixedScaleKernel.cpp
  Backend/Kernel/TgShuffleChannelKernel.cpp
  Backend/Kernel/TgSliceKernel.cpp
  Backend/Kernel/TgFixedDeconvKernel.cpp
  Backend/Kernel/TgSwapChannelKernel.cpp
  Backend/Kernel/TgTileKernel.cpp
  Backend/Kernel/TgUpsampleKernel.cpp
  Backend/Kernel/TgBf16ConcatKernel
  Backend/Kernel/TgBf16FcKernel.cpp
  Backend/Kernel/TgBf16GruKernel.cpp
  Backend/Kernel/TgBf16LstmKernel.cpp
  Backend/Kernel/TgBf16PoolingKernel.cpp
  Backend/Kernel/TgBf16PreluKernel.cpp
  Backend/Kernel/TgBf16ReluKernel.cpp
  Backend/Kernel/TgBf16ScaleKernel.cpp
  Backend/Kernel/TgBf16SoftmaxKernel.cpp
  Backend/Kernel/TgBf16SquareKernel.cpp
  Backend/Kernel/TgBf16BroadcastKernel.cpp
  Backend/Kernel/TgBf16MatMulKernel.cpp
  Backend/Kernel/TlBroadcastMul.cpp
  Backend/Kernel/TlConcat.cpp
  Backend/Kernel/TlConv.cpp
  Backend/Kernel/TlCrop.cpp
  Backend/Kernel/TlDeconv.cpp
  Backend/Kernel/TlEltwise.cpp
  Backend/Kernel/TlLeakyRelu.cpp
  Backend/Kernel/TlLrn.cpp
  Backend/Kernel/TlLut.cpp
  Backend/Kernel/TlPad.cpp
  Backend/Kernel/TlPooling.cpp
  Backend/Kernel/TlPrelu.cpp
  Backend/Kernel/TlRelu.cpp
  Backend/Kernel/TlQuant.cpp
  Backend/Kernel/TlRelu.cpp
  Backend/Kernel/TlScale.cpp
  Backend/Kernel/TlTdma.cpp
  Backend/Kernel/TlUpsample.cpp
  Backend/Kernel/TlMacConst.cpp
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
    MLIRTPUDeepFusion
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
    MLIRStandardOps
    MLIRTPUDeepFusion
    LLVMSupport
    cvikernel
    ${tpu_LINKER_LIBS}
  )
endif()

add_dependencies(MLIRTPU
  MLIRTPUOpsIncGen
  MLIRTPUInterfaceIncGen
  MLIRTPUAttributeIncGen
  MLIRIR
  LLVMSupport
)
add_subdirectory(Translate)
add_subdirectory(DeepFusion)
add_subdirectory(MemRefLowering)
