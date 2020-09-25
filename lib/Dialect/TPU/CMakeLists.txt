if(NOT tpu_LINKER_LIBS)
  set(tpu_LINKER_LIBS "")
endif(NOT tpu_LINKER_LIBS)
list(APPEND tpu_LINKER_LIBS
  mkldnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-pedantic")

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
  Backend/Kernel/cast_kernel.cpp
  Backend/Kernel/concat_bmkernel.cpp
  Backend/Kernel/conv_kernel.cpp
  Backend/Kernel/crop_bmkernel.cpp
  Backend/Kernel/mac_const_kernel.cpp
  Backend/Kernel/dilate_bmkernel.cpp
  Backend/Kernel/dma_legacy_bmkernel.cpp
  Backend/Kernel/lrn_kernel.cpp
  Backend/Kernel/lut.cpp
  Backend/Kernel/mixed_precision.cpp
  Backend/Kernel/pad_kernel.cpp
  Backend/Kernel/permute_kernel.cpp
  Backend/Kernel/pixel_shuffle_kernel.cpp
  Backend/Kernel/power_kernel.cpp
  Backend/Kernel/relu_bmkernel.cpp
  Backend/Kernel/reorg_kernel.cpp
  Backend/Kernel/scale_kernel.cpp
  Backend/Kernel/shuffle_channel_kernel.cpp
  Backend/Kernel/slice_kernel.cpp
  Backend/Kernel/src_deconv_bmkernel.cpp
  Backend/Kernel/src_tl_api.cpp
  Backend/Kernel/swap_channel_kernel.cpp
  Backend/Kernel/tensor_common.cpp
  Backend/Kernel/tile_kernel.cpp
  Backend/Kernel/tiling.cpp
  Backend/Kernel/upsample_kernel.cpp
  Backend/Kernel/tl_broadcast_mul.cpp
  Backend/Kernel/tl_concat.cpp
  Backend/Kernel/tl_conv.cpp
  Backend/Kernel/tl_crop.cpp
  Backend/Kernel/tl_deconv.cpp
  Backend/Kernel/tl_eltwise.cpp
  Backend/Kernel/tl_leaky_relu.cpp
  Backend/Kernel/tl_lrn.cpp
  Backend/Kernel/tl_lut.cpp
  Backend/Kernel/tl_pad.cpp
  Backend/Kernel/tl_pooling.cpp
  Backend/Kernel/tl_prelu.cpp
  Backend/Kernel/tl_quant.cpp
  Backend/Kernel/tl_relu.cpp
  Backend/Kernel/tl_scale.cpp
  Backend/Kernel/tl_tdma.cpp
  Backend/Kernel/tl_upsample.cpp
  Backend/Kernel/tl_mac_const.cpp
  Backend/Kernel/bf16_concat_kernel.cpp
  Backend/Kernel/bf16_fc_kernel.cpp
  Backend/Kernel/bf16_gru.cpp
  Backend/Kernel/bf16_lstm.cpp
  Backend/Kernel/bf16_pooling_kernel.cpp
  Backend/Kernel/bf16_prelu_kernel.cpp
  Backend/Kernel/bf16_relu_bmkernel.cpp
  Backend/Kernel/bf16_scale_kernel.cpp
  Backend/Kernel/bf16_softmax.cpp
  Backend/Kernel/bf16_tiling.cpp
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
