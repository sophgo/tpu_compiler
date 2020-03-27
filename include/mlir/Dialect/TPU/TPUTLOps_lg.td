//===-- TPUTLOps.td - TPU dialect operation definitions ------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines some operations of the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_TL_OPS_LG
#define TPU_TL_OPS_LG

include "mlir/Dialect/TPU/TPUBase.td"
include "mlir/Dialect/TPU/TPUInterface.td"

//===----------------------------------------------------------------------===//
// Quantization methods enum definitions.
//===----------------------------------------------------------------------===//

// Allowed quantization methods cases
// def TPU_LS_S2L : I32EnumAttrCase<"S2L", 0>;
// def TPU_LS_L2S : I32EnumAttrCase<"L2S", 1>;
// def TPU_LS_S2S : I32EnumAttrCase<"S2S", 2>;
// def TPU_LS_L2L    : I32EnumAttrCase<"L2L", 3>;
// def TPU_LS_S2TSM    : I32EnumAttrCase<"S2TSM", 4>;
// def TPU_LS_L2TSM    : I32EnumAttrCase<"L2TSM", 5>;
// def TPU_LS_TSM2S    : I32EnumAttrCase<"TSM2S", 6>;
// def TPU_LS_TSM2L    : I32EnumAttrCase<"TSM2L", 7>;

// def TPU_LSENUM:I32EnumAttr<
//     "I32EnumAttr", "TL load/store direction", [
//       TPU_LS_S2L,  TPU_LS_L2S,  TPU_LS_S2S, TPU_LS_L2L,
//       TPU_LS_S2TSM, TPU_LS_L2TSM, TPU_LS_TSM2S, TPU_LS_TSM2L
//     ]>;

def TPU_LS_S2L : StrEnumAttrCase<"S2L">;
def TPU_LS_L2S : StrEnumAttrCase<"L2S">;
def TPU_LS_S2S : StrEnumAttrCase<"S2S">;
def TPU_LS_L2L    : StrEnumAttrCase<"L2L">;
def TPU_LS_S2TSM    : StrEnumAttrCase<"S2TSM">;
def TPU_LS_L2TSM    : StrEnumAttrCase<"L2TSM">;
def TPU_LS_TSM2S    : StrEnumAttrCase<"TSM2S">;
def TPU_LS_TSM2L    : StrEnumAttrCase<"TSM2L">;

def TPU_LSDirAttr:StrEnumAttr<
    "TPU_LSDirAttr", "TL load/store direction", [
      TPU_LS_S2L,  TPU_LS_L2S,  TPU_LS_S2S, TPU_LS_L2L,
      TPU_LS_S2TSM, TPU_LS_L2TSM, TPU_LS_TSM2S, TPU_LS_TSM2L
    ]>;

def TPU_TENSOR_NEURON : StrEnumAttrCase<"NEURON">;
def TPU_TENSOR_COEFF : StrEnumAttrCase<"CONV_COEFF">;
def TPU_TENSOR_DEPTH : StrEnumAttrCase<"CONV_DEPTH_OPD1">;
def TPU_TENSOR_BIAS : StrEnumAttrCase<"BIAS">;
def TPU_TENSOR_MATRIX : StrEnumAttrCase<"MATRIX">;

def TPU_TensorTypeAttr:StrEnumAttr<
    "TPU_TensorTypeAttr", "Tensor type for load/store", [
      TPU_TENSOR_NEURON, TPU_TENSOR_COEFF,  TPU_TENSOR_DEPTH,
      TPU_TENSOR_BIAS,  TPU_TENSOR_MATRIX
    ]>;
//===----------------------------------------------------------------------===//
// TL op definitions.
//===----------------------------------------------------------------------===//

def TPU_TL_LG_Conv2DOp : TPU_Op<"tl_lg_conv_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Convolution operator";

  let description = [{
    [IMPORTANT] : for INT8_MULTIPLIER quantized convolution only.
    Performs TL convolution operation on inputs memref, and output to output
      memref. The weight are still global memory tensors, so this operator NOT has
      to handle loading weight by itself.

    Examples:
       %A = tpu.tl_load(%2) {gaddr = 0x60000000}: memref<?x?xf32, offset: 0, strides: [?, 1]>
       %B = tpu.tl_alloc() : memref<?x?xf32, offset: 0, strides: [?, 1]>
       tpu.tl_lw_conv_2d(%A, %B, %4, %5) { ... } : \
           (memref<1x64x56x56xi8, offset: 0x100, strides: [?, 1]>, \
            memref<1x256x56x56xi8, offset: 0x4000, strides: [?, 1]>, \
            tensor<256x64x3x3xi8>, tensor<256x9xi8>) -> tensor<1x256x56x56xf32>)

    Inputs:
      `inputs[0]`  : required: the input memref.
      `inputs[1]`  : required: the filter tensor.
      `inputs[2]`  : required: the per_channel_info tensor.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensor:$pc_info,
    TPU_ConvParamAttr:$param,
    DefaultValuedAttr<I32Attr, "0">:$pad_top_h,
    DefaultValuedAttr<I32Attr, "0">:$pad_bottom_h,
    DefaultValuedAttr<I32Attr, "0">:$pad_left_w,
    DefaultValuedAttr<I32Attr, "0">:$pad_right_w,
    I32Attr:$la_input,
    I32Attr:$la_filter,
    I32Attr:$la_output,
    I32Attr:$la_bias,
    I32Attr:$la_working,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TL_LG_EltwiseAddOp : TPU_Op<"tl_lg_eltwise_add",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Eltwise Add operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$addend,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32ArrayAttr>:$m_i8_inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<I32ArrayAttr>:$la_input,
    I32Attr:$la_working,
    I32Attr:$la_output,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

class TPU_TL_LG_Pool2DOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL " # opQuant # " Pool2D " # opType # " operator.";

  let description = [{
    Inputs:
      `input`          : required, the input1 activation memref.

    Attributes:
      `rshift`          : optional, rshift, an int8 value.
      `m_i8`            : optional, multiplier for output.
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, padding, and do_relu.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTLOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32Attr>:$m_i8,
    I32Attr:$la_input,
    I32Attr:$la_output,
    TPU_PoolParamAttr:$param,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TL_LG_INT8_PoolAvg2DOp : TPU_TL_LG_Pool2DOp<"tl_int8_pool_avg_2d", "Int8", "Avg">;
def TPU_TL_LG_INT8_PoolMax2DOp : TPU_TL_LG_Pool2DOp<"tl_int8_pool_max_2d", "Int8", "Max">;


def TPU_TL_LG_LoadNeuronOp : TPU_Op<"tl_lg_load_neuron",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Load operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_LSDirAttr:$ls_direction,
    OptionalAttr<NonNegativeI64Attr>:$laddr,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    TPU_TensorTypeAttr:$tensor_type,
    BoolAttr:$align,
    BoolAttr:$transpose,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TL_LG_LoadCoeffOp : TPU_Op<"tl_lg_load_coeff",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Load operator";

  let description = [{
  }];

  let arguments = (
    ins AnyMemRef:$input,
    TPU_LSDirAttr:$ls_direction,
    OptionalAttr<NonNegativeI64Attr>:$laddr,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    DefaultValuedAttr<BoolAttr, "true">:$lowered,
    TPU_TensorTypeAttr:$tensor_type,
    DefaultValuedAttr<TPU_WeightAttr, "FP32">:$storage,
    BoolAttr:$align,
    BoolAttr:$transpose,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TL_LG_StoreOp : TPU_Op<"tl_lg_store",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Store operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_LSDirAttr:$ls_direction,
    OptionalAttr<NonNegativeI64Attr>:$laddr,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    BoolAttr:$align,
    BoolAttr:$transpose,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TL_LG_JoinOp : TPU_Op<"tl_lg_join",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTLOpCodegenInterface>]> {
  let summary = "TL Join operator";

  let description = [{
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$enable_parallel,
    DefaultValuedAttr<BoolAttr, "false">:$disable_parallel,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

#endif // TPU_TL_OPS_LG
