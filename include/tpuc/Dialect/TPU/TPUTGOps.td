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

#ifndef TPU_TG_OPS
#define TPU_TG_OPS

include "tpuc/Dialect/TPU/TPUBase.td"

//
// Notes: use Tensor for the first step, later will change to memref
//

//===----------------------------------------------------------------------===//
// TPU TG op definitions.
//===----------------------------------------------------------------------===//
class TPU_TG_AbsOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Abs operator.";

  let description = [{
    Inputs:
      `input`          : required, the variadic activation memref.

    Attributes:
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_AbsOp : TPU_TG_AbsOp<"tg_int8_abs", "Int8">;
def TPU_TG_BF16_AbsOp : TPU_TG_AbsOp<"tg_bf16_abs", "Bf16">;


class TPU_TG_ConcatOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Concat operator.";

  let description = [{
    Inputs:
      `inputs`          : required, the variadic activation memref.

    Attributes:
      `axis`            : required, the axis the concat is applying
      `rshift`          : optional, rshift, an int8 value.
      `m_i8_inputs`     : optional, multipliers for inputs. one int8 value for
                          each input (saved as I32ArrayAttr)
      `m_i8_output`     : optional, multiplier for output.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32ArrayAttr>:$m_i8_inputs,
    OptionalAttr<I8Attr>:$m_i8_output,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    I32Attr:$axis,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ConcatOp : TPU_TG_ConcatOp<"tg_int8_concat", "Int8">;
def TPU_TG_BF16_ConcatOp : TPU_TG_ConcatOp<"tg_bf16_concat", "Bf16">;

def TPU_TG_ConcatNOp: TPU_Op<"tg_concat_n",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Concat operator.";

  let description = [{
    Inputs:
      `inputs`          : required, the variadic activation memref.

    Attributes:
      `axis`            : required, the axis the concat is applying
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I32Attr:$axis,
    OptionalAttr<I64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

class TPU_TG_ConvOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Convolution operator";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
      `filter`          : required, the filter weight memref.
      `pc_info`         : required, the perchannel weight memref.

    Attributes:
      `pt_rshift`       : optional, rshift for per-tensor mode.
      `param`           : required, a ConvParam struct attributes, carrying
                          stride, padding, dilation, group, is_dw, with_bias,
                          and do_relu.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[I32,I16,I8]>:$pc_info,
    OptionalAttr<I8Attr>:$pt_rshift,
    TPU_ConvParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<BoolAttr>:$do_ic_alignment,
    OptionalAttr<F32Attr>:$negative_slope,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    DefaultValuedAttr<BoolAttr, "false">:$do_leaky_relu,
    OptionalAttr<BoolAttr>:$load_compr_act,
    OptionalAttr<BoolAttr>:$store_compr_act,
    OptionalAttr<TPU_ConvTileParamAttr>:$tile_param,
    OptionalAttr<BoolAttr>:$compressed_weight,
    OptionalAttr<TPU_ActCmprParamAttr>:$store_compr_act_param,
    OptionalAttr<TPU_ActCmprParamAttr>:$load_compr_act_param,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PT_Conv2DOp : TPU_TG_ConvOp<"tg_int8_pt_conv_2d", "Int8 Per-Tensor">;
def TPU_TG_INT8_PC_Conv2DOp : TPU_TG_ConvOp<"tg_int8_pc_conv_2d", "Int8 Per-Channel">;
def TPU_TG_BF16_Conv2DOp : TPU_TG_ConvOp<"tg_bf16_conv_2d", "Bf16">;
def TPU_TG_INT8_PT_DeConv2DOp : TPU_TG_ConvOp<"tg_int8_pt_deconv_2d", "Int8 Per-Tensor">;
def TPU_TG_INT8_PC_DeConv2DOp : TPU_TG_ConvOp<"tg_int8_pc_deconv_2d", "Int8 Per-Channel">;
def TPU_TG_BF16_DeConv2DOp : TPU_TG_ConvOp<"tg_bf16_deconv_2d", "Bf16">;



class TPU_TG_Conv3dOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Convolution operator";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
      `filter`          : required, the filter weight memref.
      `pc_info`         : required, the perchannel weight memref.

    Attributes:
      `pt_rshift`       : optional, rshift for per-tensor mode.
      `param`           : required, a ConvParam struct attributes, carrying
                          stride, padding, dilation, group, is_dw, with_bias,
                          and do_relu.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[I32,I16,I8]>:$pc_info,
    OptionalAttr<I8Attr>:$pt_rshift,
    TPU_Conv3dParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<BoolAttr>:$do_ic_alignment,
    OptionalAttr<F32Attr>:$negative_slope,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    DefaultValuedAttr<BoolAttr, "false">:$do_leaky_relu,
    OptionalAttr<BoolAttr>:$load_compr_act,
    OptionalAttr<BoolAttr>:$store_compr_act,
    OptionalAttr<TPU_ConvTileParamAttr>:$tile_param,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_BF16_Conv3DOp : TPU_TG_Conv3dOp<"tg_bf16_conv_3d", "Bf16">;

// underlying we are using conv to do broadcast mul
def TPU_TG_INT8_BroadcastMulOp : TPU_TG_ConvOp<"tg_int8_broadcast_mul", "Int8">;
def TPU_TG_BF16_BroadcastMulOp : TPU_TG_ConvOp<"tg_bf16_broadcast_mul", "Bf16">;


class TPU_TG_CropOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Crop operator.";

  let description = [{
    Inputs:
      `input`          : required, the activation memref.

    Attributes:
      `crop_shape`      : required, crop shape(saved as I32ArrayAttr).
      `crop_offset`     : required, crop offset(saved as I32ArrayAttr).
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32ArrayAttr>:$crop_shape,
    OptionalAttr<I32ArrayAttr>:$crop_offset,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_CropOp : TPU_TG_CropOp<"tg_int8_crop", "Int8">;
def TPU_TG_BF16_CropOp : TPU_TG_CropOp<"tg_bf16_crop", "Bf16">;


class TPU_TG_DilateOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Dilate operator.";

  let description = [{
    Inputs:
      `input`          : required, the activation memref.

    Attributes:
      `fill_constant`   : required, fill the dilated value
      `ins`             : required, ins[0] means ins_w, ins[1] means ins_h
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "0">:$fill_constant,
    OptionalAttr<I32ArrayAttr>:$ins,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_DilateOp : TPU_TG_DilateOp<"tg_int8_Dilate", "Int8">;
def TPU_TG_BF16_DilateOp : TPU_TG_DilateOp<"tg_bf16_Dilate", "Bf16">;


class TPU_TG_EltwiseOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Eltwise " # opType # " operator.";

  let description = [{
    Inputs:
      `inputs`          : required, the variadic activation memref.

    Attributes:
      `rshift`          : optional, rshift, an int8 value.
      `m_i8_inputs`     : optional, multipliers for inputs. one int8 value for
                          each input (saved as I32ArrayAttr)
      `m_i8_output`     : optional, multiplier for output.
      `m_i32_output`    : optional, multiplier for output, .
      `do_relu`         : required, if need to preform relu on result.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `input_offset`    : optional, inputs offset
      `output_offset`   : optional, output offset
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32ArrayAttr>:$m_i8_inputs,
    OptionalAttr<I8Attr>:$m_i8_output,
    OptionalAttr<I32Attr>:$m_i32_output,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BoolAttr, "false">:$do_early_stride,
    DefaultValuedAttr<I32Attr, "1">:$early_stride_h,
    DefaultValuedAttr<I32Attr, "1">:$early_stride_w,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I32ArrayAttr>:$input_offset,
    OptionalAttr<I32Attr>:$output_offset,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<F32ArrayAttr>:$coeff,
    OptionalAttr<BoolAttr>:$load_compr_act,
    OptionalAttr<BoolAttr>:$store_compr_act,
    OptionalAttr<TPU_ActCmprParamAttr>:$store_compr_act_param,
    OptionalAttr<TPU_ActCmprParamAttr>:$load_compr_act_param,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_EltwiseAddOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_add", "Int8", "Add">;
def TPU_TG_INT8_EltwiseMaxOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_max", "Int8", "Max">;
def TPU_TG_INT8_EltwiseMinOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_min", "Int8", "Min">;
def TPU_TG_INT8_EltwiseMulOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_mul", "Int8", "Mul">;
def TPU_TG_BF16_EltwiseAddOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_add", "Bf16", "Add">;
def TPU_TG_BF16_EltwiseMaxOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_max", "Bf16", "Max">;
def TPU_TG_BF16_EltwiseMinOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_min", "Bf16", "Min">;
def TPU_TG_BF16_EltwiseMulOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_mul", "Bf16", "Mul">;

class TPU_TG_BroadcastOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Broadcast " # opType # " operator.";

  let description = [{
    Inputs:
      `inputs`          : required, the variadic activation memref.

    Attributes:
      `rshift`          : optional, rshift, an int8 value.
      `m_i8_inputs`     : optional, multipliers for inputs. one int8 value for
                          each input (saved as I32ArrayAttr)
      `m_i8_output`     : optional, multiplier for output.
      `m_i32_output`    : optional, multiplier for output, .
      `do_relu`         : required, if need to preform relu on result.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32ArrayAttr>:$m_i8_inputs,
    OptionalAttr<I8Attr>:$m_i8_output,
    OptionalAttr<I32Attr>:$m_i32_output,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<BoolAttr>:$load_compr_act,
    OptionalAttr<BoolAttr>:$store_compr_act,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_BroadcastAddOp : TPU_TG_BroadcastOp<"tg_int8_broadcast_add", "Int8", "Add">;
def TPU_TG_INT8_BroadcastSubOp : TPU_TG_BroadcastOp<"tg_int8_broadcast_sub", "Int8", "Sub">;
def TPU_TG_BF16_BroadcastAddOp : TPU_TG_BroadcastOp<"tg_bf16_broadcast_add", "Bf16", "Add">;
def TPU_TG_BF16_BroadcastSubOp : TPU_TG_BroadcastOp<"tg_bf16_broadcast_sub", "Bf16", "Sub">;

class TPU_TG_FullyConnectedOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " FullyConnected operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `do_relu`         : required, whether to do relu on result.
      `rshift`          : optional, rshift for positive, an int8 value.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[I32, I16]>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32Attr>:$mutliplier,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<TPU_FcTileParamAttr>:$tile_param,
    OptionalAttr<BoolAttr>:$compressed_weight,
    OptionalAttr<I32ArrayAttr>:$compr_weight_poss,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_FullyConnectedOp : TPU_TG_FullyConnectedOp<"tg_int8_fully_connected", "Int8">;
def TPU_TG_BF16_FullyConnectedOp : TPU_TG_FullyConnectedOp<"tg_bf16_fully_connected", "Bf16">;

class TPU_TG_InterpOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Interp operator.";
  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
    Attributes:
      `rshift_pos`      : optional, rshift for positive, an int8 value.
      `m_i8_pos`        : optional, multiplier for positive, an int8 value.
      `rshift_neg`      : required, rshift for negative, an int8 value.
      `m_i8_neg`        : required, multiplier for negative, an int8 value.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `height`          : optional, height of output
      `width`           : optional, width of output
      `shrink_factor`   : optional, shrink h/w to small one, leverage conv
      `zoom_factor`     : optional, enlarge h/w to larger one, leverage deconv
      `pad_beg`         : optional, padding twice in h/w
      `pad_end`         : optional, padding twice in h/w
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    DefaultValuedAttr<I32Attr, "0">:$height,
    DefaultValuedAttr<I32Attr, "0">:$width,
    DefaultValuedAttr<I32Attr, "0">:$shrink_factor,
    DefaultValuedAttr<I32Attr, "0">:$zoom_factor,
    DefaultValuedAttr<I32Attr, "0">:$pad_beg,
    DefaultValuedAttr<I32Attr, "0">:$pad_end,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_INT8_InterpOp : TPU_TG_InterpOp<"tg_int8_interp", "Int8">;
def TPU_TG_BF16_InterpOp : TPU_TG_InterpOp<"tg_bf16_interp", "Bf16">;

class TPU_TG_GruOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " GRU operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
      `weight`          : required, parameter weight memref for update, reset, and hidden gates.
      `recurrence`       : required, recurrence weight memref for update, reset, and hidden gates.
      `bias`            : optional, bias vectors memref for update, reset, and hidden gates.
      `initial_h`       : required, initial value of the hidden.
      'sigmoid_table'           : required, the sigmoid_table memref.
      'sigmoid_slope_table'           : required, the sigmoid_slope_table memref.
      'tanh_table'           : required, the tanh_table memref.
      'tanh_slope_table'           : required, the tanh_slope_table memref.


    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$recurrence,
    TPU_TensorOfOrNone<[I32, F32, BF16, I16, I8]>:$bias,
    AnyTensor:$initial_h,
    AnyTensor:$sigmoid_table,
    AnyTensor:$sigmoid_slope_table,
    AnyTensor:$tanh_table,
    AnyTensor:$tanh_slope_table,
    DefaultValuedAttr<BoolAttr, "true">:$linear_before_reset,
    DefaultValuedAttr<BoolAttr, "false">:$bidirectional,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_INT8_GruOp : TPU_TG_GruOp<"tg_int8_gru", "Int8">;
def TPU_TG_BF16_GruOp : TPU_TG_GruOp<"tg_bf16_gru", "Bf16">;

class TPU_TG_LstmOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Lstm operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
      `weight`          : required, parameter weight memref for update, reset, and hidden gates.
      `recurrence`       : required, recurrence weight memref for update, reset, and hidden gates.
      `bias`            : optional, bias vectors memref for update, reset, and hidden gates.
      `initial_h`       : required, initial value of the hidden.
      `initial_c`       : required, initial value of the hidden.
      'sigmoid_table'           : required, the sigmoid_table memref.
      'sigmoid_slope_table'           : required, the sigmoid_slope_table memref.
      'tanh_table'           : required, the tanh_table memref.
      'tanh_slope_table'           : required, the tanh_slope_table memref.


    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$recurrence,
    TPU_TensorOfOrNone<[I32, F32, BF16, I16, I8]>:$bias,
    AnyTensor:$initial_h,
    AnyTensor:$initial_c,
    AnyTensor:$sigmoid_table,
    AnyTensor:$sigmoid_slope_table,
    AnyTensor:$tanh_table,
    AnyTensor:$tanh_slope_table,
    DefaultValuedAttr<BoolAttr, "false">:$bidirectional,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_INT8_LstmOp : TPU_TG_LstmOp<"tg_int8_lstm", "Int8">;
def TPU_TG_BF16_LstmOp : TPU_TG_LstmOp<"tg_bf16_lstm", "Bf16">;

class TPU_TG_LeakyReluOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " LeakyRelu operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `rshift_pos`      : optional, rshift for positive, an int8 value.
      `m_i8_pos`        : optional, multiplier for positive, an int8 value.
      `rshift_neg`      : required, rshift for negative, an int8 value.
      `m_i8_neg`        : required, multiplier for negative, an int8 value.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `input_offset`    : optional, in asymmetric, add input_offset to input tensor before do leaky relu.
      `output_offset`    : optional, in asymmetric, add output_offset at final.
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<F32Attr>:$negative_slope,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    OptionalAttr<I32Attr>:$input_offset,
    OptionalAttr<I32Attr>:$output_offset,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_LeakyReluOp : TPU_TG_LeakyReluOp<"tg_int8_leaky_relu", "Int8">;
def TPU_TG_BF16_LeakyReluOp : TPU_TG_LeakyReluOp<"tg_bf16_leaky_relu", "Bf16">;

class TPU_TG_LrnOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " LRN operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.
      `sqr_lut`         : required, sqr lut.
      `power_lut`       : required, power lut.

    Attributes:
      `local_size`      : required, accross channel local size
      `sum_rshift`      : required, sum right shift width
      `lrn_rshift`      : required, lrn right shift width
      `quant_data0`     : required, quantilization x [0]
      `quant_data1`     : required, quantilization x [1]
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$sqr_lut,
    AnyTensor:$power_lut,
    I32Attr:$local_size,
    I32Attr:$sum_rshift,
    I32Attr:$lrn_rshift,
    I32Attr:$quant_data0,
    I32Attr:$quant_data1,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$k,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_LrnOp : TPU_TG_LrnOp<"tg_int8_lrn", "Int8">;
def TPU_TG_BF16_LrnOp : TPU_TG_LrnOp<"tg_bf16_lrn", "Bf16">;

class TPU_TG_LutOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # "lut operator.";

  let description = [{
    Inputs:
      `input`          : required, the variadic activation memref.
      `table`          : required, the lookup table

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `method`          : optional(bf16 required)lookup table method, mantissa or slope
      `max_range`       : reauired, we cut to max_range and outlier saturate it
      `min_range`       : reauired, we cut to min_range and outlier saturate it
      `added_offset`    : optional, set to ture if we had added offset value via add op
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$table,
    AnyTensor:$table_mantissa,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$method,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    DefaultValuedAttr<BoolAttr, "false">:$added_offset,
    DefaultValuedAttr<I32Attr, "-1">:$layer_id,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_LutOp : TPU_TG_LutOp<"tg_int8_lut", "Int8">;
def TPU_TG_BF16_LutOp : TPU_TG_LutOp<"tg_bf16_lut", "BF16">;


class TPU_TG_PermuteOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Permute operator.";

  let description = [{
    Inputs:
      `input`          : required, the activation memref.

    Attributes:
      `order0`          : required, The new order0 of the axes of data.
      `order1`          : required, The new order1 of the axes of data.
      `order2`          : required, The new order2 of the axes of data.
      `order3`          : required, The new order3 of the axes of data.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$order0,
    I32Attr:$order1,
    I32Attr:$order2,
    I32Attr:$order3,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PermuteOp : TPU_TG_PermuteOp<"tg_int8_permute", "Int8">;
def TPU_TG_BF16_PermuteOp : TPU_TG_PermuteOp<"tg_bf16_permute", "Bf16">;

class TPU_TG_Pool2DOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Pool2D " # opType # " operator.";

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
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I8Attr>:$m_i8,
    TPU_PoolParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PoolAvg2DOp : TPU_TG_Pool2DOp<"tg_int8_pool_avg_2d", "Int8", "Avg">;
def TPU_TG_INT8_PoolMax2DOp : TPU_TG_Pool2DOp<"tg_int8_pool_max_2d", "Int8", "Max">;
def TPU_TG_BF16_PoolAvg2DOp : TPU_TG_Pool2DOp<"tg_bf16_pool_avg_2d", "Bf16", "Avg">;
def TPU_TG_BF16_PoolMax2DOp : TPU_TG_Pool2DOp<"tg_bf16_pool_max_2d", "Bf16", "Max">;

class TPU_TG_Pool3DOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Pool3D " # opType # " operator.";

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
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I8Attr>:$m_i8,
    TPU_Pool3dParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_INT8_PoolMax3DOp : TPU_TG_Pool3DOp<"tg_int8_pool_max_3d", "Int8", "Max">;
def TPU_TG_BF16_PoolMax3DOp : TPU_TG_Pool3DOp<"tg_bf16_pool_max_3d", "Bf16", "Max">;

class TPU_TG_PReluOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Prelu operator.";

  let description = [{
    Inputs:
      `input`          : required, the variadic activation memref.
      `negative_slope` : required, negative_slope value memref.

    Attributes:
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `name`            : required, name for comparing, or debug.
      `rshift_pos`     : optional, rshift for positive, an int8 value.
      `m_i8_pos`       : optional, multiplier for positive, an int8 value.
      `rshift_neg`     : required, rshift for negative, an int8 value.
      `chipname`       : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$negative_slope,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PReluOp : TPU_TG_PReluOp<"tg_int8_prelu", "Int8">;
def TPU_TG_BF16_PReluOp : TPU_TG_PReluOp<"tg_bf16_prelu", "Bf16">;

def TPU_TG_QuantOp : TPU_Op<"tg_quant",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Qaunt/Dequant operator.";

  let description = [{
    Inputs:
      `input`          : required, the input1 activation memref.

    Attributes:
      `from`           : required, the source type to be dequantized/quantized.
      `to`             : required, the destination type that quantize/dequantize to.
      `threshold`      : optional, scale = threshold/128.0 (from int8) or 128/threshold.0 (to int8);
                         if none, scale = 1.0
      `zero_point`         : optional, offset, asymmetric zero_point
      `region`         : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantModeAttr:$from,
    TPU_QuantModeAttr:$to,
    F32Attr:$scale,
    OptionalAttr<I32Attr>:$zero_point,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_ReQuantOp : TPU_Op<"tg_requant",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " ReQuant operator.";

  let description = [{
    Inputs:
      `input`          : required, the input activation memref.

    Attributes:
      `input_offset`   : required, input_offset
      `output_offset`  : required, output_offset
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<F32Attr>:$qscale,
    OptionalAttr<I32Attr>:$input_offset,
    OptionalAttr<I32Attr>:$output_offset,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

class TPU_TG_ReluOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Relu operator.";

  let description = [{
    Inputs:
      `input`          : required, the variadic activation memref.

    Attributes:
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReluOp : TPU_TG_ReluOp<"tg_int8_relu", "Int8">;
def TPU_TG_BF16_ReluOp : TPU_TG_ReluOp<"tg_bf16_relu", "Bf16">;


class TPU_TG_ReorgOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Reorg operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `stride`          : required, channel / height /width stride.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$stride,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReorgOp : TPU_TG_ReorgOp<"tg_int8_reorg", "Int8">;
def TPU_TG_BF16_ReorgOp : TPU_TG_ReorgOp<"tg_bf16_reorg", "Bf16">;

class TPU_TG_ReverseOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Reverse operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `axis`           : required, reverse dim
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$axis,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReverseOp : TPU_TG_ReverseOp<"tg_int8_reverse", "Int8">;
def TPU_TG_BF16_ReverseOp : TPU_TG_ReverseOp<"tg_bf16_reverse", "Bf16">;

class TPU_TG_ShuffleChannelOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " ShuffleChannel operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `group`           : required, channel group of shuffle.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$group,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ShuffleChannelOp : TPU_TG_ShuffleChannelOp<"tg_int8_shufflechannel", "Int8">;
def TPU_TG_BF16_ShuffleChannelOp : TPU_TG_ShuffleChannelOp<"tg_bf16_shufflechannel", "Bf16">;


class TPU_TG_TileComOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " PixelShuffle operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `resp`            : optional, `input` is promoted to be d-dimensional by prepending new axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not the desired behavior, promote `input` to d-dimensions manually before calling this function.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32ArrayAttr>:$resp,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_TileOp : TPU_TG_TileComOp<"tg_int8_tile", "Int8">;
def TPU_TG_BF16_TileOp : TPU_TG_TileComOp<"tg_bf16_tile", "Bf16">;
def TPU_TG_INT8_TileInterpOp : TPU_TG_TileComOp<"tg_int8_tile_interp", "Int8">;
def TPU_TG_BF16_TileInterpOp : TPU_TG_TileComOp<"tg_bf16_tile_interp", "Bf16">;

class TPU_TG_PixelShuffleOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " PixelShuffle operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `upscale_factor`  : required, indicate for upsample upscale_factor.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$upscale_factor,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    DefaultValuedAttr<StrAttr, "CRD">:$mode,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PixelShuffleOp : TPU_TG_PixelShuffleOp<"tg_int8_pixelshuffle", "Int8">;
def TPU_TG_BF16_PixelShuffleOp : TPU_TG_PixelShuffleOp<"tg_bf16_pixelshuffle", "Bf16">;

class TPU_TG_ClipOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Clip operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `max`: optional, Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape).
      `min`: optional, Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape).

      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<F32Attr, "999">:$min,
    DefaultValuedAttr<F32Attr, "-999">:$max,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ClipOp : TPU_TG_ClipOp<"tg_int8_clip", "Int8">;
def TPU_TG_BF16_ClipOp : TPU_TG_ClipOp<"tg_bf16_clip", "Bf16">;

class TPU_TG_SliceOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " slice operator.";

  let description = [{
    Slices an tensor to multiple output tensors along a given dimension
    (currently channel only) with given slice indices.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `axis`            : required, the axis to slice.
      `offset`          : required, the offset the output is sliced within the
                          input, along the specified axis.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    I32Attr:$offset,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<BoolAttr>:$gaddr_updated,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_SliceOp : TPU_TG_SliceOp<"tg_int8_slice", "Int8">;
def TPU_TG_BF16_SliceOp : TPU_TG_SliceOp<"tg_bf16_slice", "Bf16">;

class TPU_TG_SoftmaxOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " softmax operator.";

  let description = [{
    Perform softmax on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `exponential_table`           : required, the exponential table.
      `exponential_slope_table`           : required, the exponential slope table.
      `reciprocal_table`           : required, the reciprocal table.
      `reciprocal_mantissa_table`           : required, the reciprocal mantissa table.

    Attributes:
      `axis`            : optional, the axis for softmax to perform.
      `name`            : required, name for calibration, comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$exponential_table,
    AnyTensor:$exponential_slope_table,
    AnyTensor:$reciprocal_table,
    AnyTensor:$reciprocal_mantissa_table,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<BoolAttr>:$gaddr_updated,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_SoftmaxOp : TPU_TG_SoftmaxOp<"tg_int8_softmax", "Int8">;
def TPU_TG_BF16_SoftmaxOp : TPU_TG_SoftmaxOp<"tg_bf16_softmax", "Bf16">;

class TPU_TG_SwapChannelOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " SwapChannel operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32ArrayAttr>:$channel_order,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_SwapChannelOp : TPU_TG_SwapChannelOp<"tg_int8_swapchannel", "Int8">;
def TPU_TG_BF16_SwapChannelOp : TPU_TG_SwapChannelOp<"tg_bf16_swapchannel", "Bf16">;


class TPU_TG_UpsampleOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Upsample operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `scale`           : required, scale for upsampling.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$scale_h,
    I32Attr:$scale_w,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_UpsampleOp : TPU_TG_UpsampleOp<"tg_int8_upsample", "Int8">;
def TPU_TG_BF16_UpsampleOp : TPU_TG_UpsampleOp<"tg_bf16_upsample", "Bf16">;

class TPU_TG_GenericTpuOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " LeakyRelu operator.";

  let description = [{
    Generic Tpu Op.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required
      `name`            : required
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.
  }];

  let arguments = (
    ins Variadic<TPU_TensorOfOrNone<[F32, BF16, I8]>>:$inputs,
    StrAttr:$name,
    StrAttr:$operation_name,
    DictionaryAttr:$param,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_GenericTpuOp : TPU_TG_GenericTpuOp<"tg_int8_generic_tpu_op", "Int8">;
def TPU_TG_BF16_GenericTpuOp : TPU_TG_GenericTpuOp<"tg_bf16_generic_tpu_op", "Bf16">;

def TPU_GenericCpuOp : TPU_Op<"generic_cpu_op",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "generic cpu operator";

  let description = [{
    Generic Cpu Op.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required
      `name`            : required
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

  }];

  let arguments = (
    ins Variadic<TPU_TensorOfOrNone<[F32, BF16, AnyInteger]>>:$inputs,
    StrAttr:$name,
    StrAttr:$operation_name,
    DictionaryAttr:$param,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_CallOp : TPU_Op <"tg_call",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]>{
  let summary = "Call a function.";
  let description = [{
    The "call" operation represents a direct call to a function that is within
    the same symbol scope as the call.  The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    function attribute named "callee".

      %2 = tpu.tg_call @my_add(%0, %1) : (f32, f32) -> f32
  }];
  let arguments = (ins
    OptionalAttr<I64Attr>:$gaddr,
    StrAttr:$name,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments,
    OptionalAttr<StrAttr>:$chipname
    );
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let results = (outs Variadic<AnyType>);
}

class TPU_TG_PadOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Pad operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `scale`           : required, scale for upsampling.
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `pads`            : tensor containing the number of start and end
                          pad values for axis
      `const_val`       : A scalar value to be used if the mode
                          chosen is `constant` (by default it is 0).
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<I32ArrayAttr>:$pads,
    DefaultValuedAttr<F32Attr, "0">:$const_val,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PadOp : TPU_TG_PadOp<"tg_int8_pad", "Int8">;
def TPU_TG_BF16_PadOp : TPU_TG_PadOp<"tg_bf16_pad", "Bf16">;

class TPU_TG_ReduceMeanOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Reduce mean operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `axes`            : along which axis to do reduce
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I8Attr>:$m_i8,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<I32ArrayAttr>:$axes,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReduceMeanOp : TPU_TG_ReduceMeanOp<"tg_int8_reduce_mean", "Int8">;
def TPU_TG_BF16_ReduceMeanOp : TPU_TG_ReduceMeanOp<"tg_bf16_reduce_mean", "Bf16">;

class TPU_TG_ReduceMaxOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Reduce max operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `axes`            : along which axis to do reduce
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I8Attr>:$m_i8,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<I32ArrayAttr>:$axes,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReduceMaxOp : TPU_TG_ReduceMaxOp<"tg_int8_reduce_max", "Int8">;
def TPU_TG_BF16_ReduceMaxOp : TPU_TG_ReduceMaxOp<"tg_bf16_reduce_max", "Bf16">;

class TPU_TG_SquareOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Square operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_BF16_SquareOp : TPU_TG_SquareOp<"tg_bf16_square", "Bf16">;

class TPU_TG_QuadraticSumOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Sum of Squares operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    OptionalAttr<BoolAttr>:$high_precision,
    StrAttr:$name,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_BF16_QuadraticSumOp : TPU_TG_QuadraticSumOp<"tg_bf16_quadratic_sum", "Bf16">;

class TPU_TG_MatMulOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Matrix Multipy operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.
      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_MatMulOp : TPU_TG_MatMulOp<"tg_int8_matmul", "Int8">;
def TPU_TG_BF16_MatMulOp : TPU_TG_MatMulOp<"tg_bf16_matmul", "Bf16">;

class TPU_TG_Yuv420CscOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " Yuv420 operator.";

  let description = [{
    Inputs:
      `input`           : required, the input activation memref.

    Attributes:
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `gaddr`           : optional, address in global memory, normally the
                                     offset within the memory region.
      `buffer_reused`   : optional, whether buffer is reused.

      `name`            : required, name for comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32ArrayAttr>:$channel_order,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_Yuv420CscOp : TPU_TG_Yuv420CscOp<"tg_int8_yuv420_csc", "Int8">;

class TPU_TG_ZeroMaskOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "zero mask operator";

  let description = [{
    if input[x] < 0, output[x] = 0; if input[x] = 0, output[x] = 1

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `chipname`        : optional, chipname.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<I64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$chipname
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TG_INT8_ZeroMaskOp : TPU_TG_ZeroMaskOp<"tg_int8_zero_mask", "Int8">;
def TPU_TG_BF16_ZeroMaskOp : TPU_TG_ZeroMaskOp<"tg_bf16_zero_mask", "Bf16">;

#endif // TPU_TG_OPS
