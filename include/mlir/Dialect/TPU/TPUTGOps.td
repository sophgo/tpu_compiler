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

include "mlir/Dialect/TPU/TPUBase.td"

//
// Notes: use Tensor for the first step, later will change to memref
//

//===----------------------------------------------------------------------===//
// TPU TG op definitions.
//===----------------------------------------------------------------------===//

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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    I32Attr:$axis,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ConcatOp : TPU_TG_ConcatOp<"tg_int8_concat", "Int8">;
def TPU_TG_BF16_ConcatOp : TPU_TG_ConcatOp<"tg_bf16_concat", "Bf16">;


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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.
      `fuse_prev`       : optional, if present, be fused with the prev TG op,
                          and current Op shouldn't generate any code anymore.
      `fuse_next`       : optional, if present, fuse with the next TG op.

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
    AnyTensor:$pc_info,
    OptionalAttr<I8Attr>:$pt_rshift,
    TPU_ConvParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_prev,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_next
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PT_Conv2DOp : TPU_TG_ConvOp<"tg_int8_pt_conv_2d", "Int8 Per-Tensor">;
def TPU_TG_INT8_PC_Conv2DOp : TPU_TG_ConvOp<"tg_int8_pc_conv_2d", "Int8 Per-Channel">;
def TPU_TG_BF16_Conv2DOp : TPU_TG_ConvOp<"tg_bf16_conv_2d", "Bf16">;
def TPU_TG_INT8_PT_DeConv2DOp : TPU_TG_ConvOp<"tg_int8_pt_deconv_2d", "Int8 Per-Tensor">;
def TPU_TG_INT8_PC_DeConv2DOp : TPU_TG_ConvOp<"tg_int8_pc_deconv_2d", "Int8 Per-Channel">;
def TPU_TG_BF16_DeConv2DOp : TPU_TG_ConvOp<"tg_bf16_deconv_2d", "Bf16">;
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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_CropOp : TPU_TG_CropOp<"tg_int8_crop", "Int8">;
def TPU_TG_BF16_CropOp : TPU_TG_CropOp<"tg_bf16_crop", "Bf16">;


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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_EltwiseAddOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_add", "Int8", "Add">;
def TPU_TG_INT8_EltwiseMaxOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_max", "Int8", "Max">;
def TPU_TG_INT8_EltwiseMulOp : TPU_TG_EltwiseOp<"tg_int8_eltwise_mul", "Int8", "Mul">;
def TPU_TG_BF16_EltwiseAddOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_add", "Bf16", "Add">;
def TPU_TG_BF16_EltwiseMaxOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_max", "Bf16", "Max">;
def TPU_TG_BF16_EltwiseMulOp : TPU_TG_EltwiseOp<"tg_bf16_eltwise_mul", "Bf16", "Mul">;


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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_FullyConnectedOp : TPU_TG_FullyConnectedOp<"tg_int8_fully_connected", "Int8">;
def TPU_TG_BF16_FullyConnectedOp : TPU_TG_FullyConnectedOp<"tg_bf16_fully_connected", "Bf16">;


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
      `gaddr`           : optional, address in global memory, normally the
                          offset within the memory region.
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.
      `fuse_prev`       : optional, if present, be fused with the prev TG op,
                          and current Op shouldn't generate any code anymore.
      `fuse_next`       : optional, if present, fuse with the next TG op.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    F32Attr:$negative_slope,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_prev,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_next
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_LeakyReluOp : TPU_TG_LeakyReluOp<"tg_int8_leaky_relu", "Int8">;
def TPU_TG_BF16_LeakyReluOp : TPU_TG_LeakyReluOp<"tg_bf16_leaky_relu", "Bf16">;

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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_LutOp : TPU_TG_LutOp<"tg_int8_lut", "Int8">;
def TPU_TG_BF16_Sqrt_LutOp : TPU_TG_LutOp<"tg_bf16_Sqrt_lut", "BF16">;
def TPU_TG_BF16_Div_LutOp : TPU_TG_LutOp<"tg_bf16_Div_lut", "BF16">;


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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    NonNegativeI32Attr:$order0,
    NonNegativeI32Attr:$order1,
    NonNegativeI32Attr:$order2,
    NonNegativeI32Attr:$order3,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PoolAvg2DOp : TPU_TG_Pool2DOp<"tg_int8_pool_avg_2d", "Int8", "Avg">;
def TPU_TG_INT8_PoolMax2DOp : TPU_TG_Pool2DOp<"tg_int8_pool_max_2d", "Int8", "Max">;
def TPU_TG_BF16_PoolAvg2DOp : TPU_TG_Pool2DOp<"tg_bf16_pool_avg_2d", "Bf16", "Avg">;
def TPU_TG_BF16_PoolMax2DOp : TPU_TG_Pool2DOp<"tg_bf16_pool_max_2d", "Bf16", "Max">;


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
      `region`          : required, the memory region to reside in, could be
                          one of `INPUT`, `OUTPUT`, `ACTIVATION` or `WEIGHT`.
                          (backend does not support `INPUT`, `OUTPUT` for now)
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.
      `rshift_pos`     : optional, rshift for positive, an int8 value.
      `m_i8_pos`       : optional, multiplier for positive, an int8 value.
      `rshift_neg`     : required, rshift for negative, an int8 value.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_PReluOp : TPU_TG_PReluOp<"tg_int8_prelu", "Int8">;
def TPU_TG_BF16_PReluOp : TPU_TG_PReluOp<"tg_bf16_prelu", "Bf16">;


class TPU_TG_ReshapeOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " reshape operator.";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
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
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ReshapeOp : TPU_TG_ReshapeOp<"tg_int8_reshape", "Int8">;
def TPU_TG_BF16_ReshapeOp : TPU_TG_ReshapeOp<"tg_bf16_reshape", "Bf16">;


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

      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    NonNegativeI32Attr:$group,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_ShuffleChannelOp : TPU_TG_ShuffleChannelOp<"tg_int8_shufflechannel", "Int8">;
def TPU_TG_BF16_ShuffleChannelOp : TPU_TG_ShuffleChannelOp<"tg_bf16_shufflechannel", "Bf16">;

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

      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

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
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_SliceOp : TPU_TG_SliceOp<"tg_int8_slice", "Int8">;
def TPU_TG_BF16_SliceOp : TPU_TG_SliceOp<"tg_bf16_slice", "Bf16">;


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
      `name`            : required, name for comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`    : support common TPU TG Op interface.
      `TpuTGOpCodegenInterface` : support generate TPU instuctions.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$scale,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}
def TPU_TG_INT8_UpsampleOp : TPU_TG_UpsampleOp<"tg_int8_upsample", "Int8">;
def TPU_TG_BF16_UpsampleOp : TPU_TG_UpsampleOp<"tg_bf16_upsample", "Bf16">;

#endif // TPU_TG_OPS
