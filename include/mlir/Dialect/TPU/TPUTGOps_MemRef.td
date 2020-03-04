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
// Defines some operations of the TPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_TG_OPS_MEMREF
#define TPU_TG_OPS_MEMREF

include "mlir/Dialect/TPU/TPUBase.td"

class TPU_TG_MemRef_ConvOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
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
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$filter,
    MemRefOf<[I32,I16,I8]>:$pc_info,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    OptionalAttr<I8Attr>:$pt_rshift,
    TPU_ConvParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_prev,
    DefaultValuedAttr<BoolAttr, "false">:$fuse_next
  );
}
def TPU_TG_MemRef_INT8_PC_Conv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_int8_pc_conv_2d", "Int8 Per-Channel">;

class TPU_TG_MemRef_EltwiseOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
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
    ins Variadic<MemRefOf<[AnyFloat, AnyInteger]>>:$inputs,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
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
}
def TPU_TG_MemRef_INT8_EltwiseAddOp : TPU_TG_MemRef_EltwiseOp<"tg_memref_int8_eltwise_add", "Int8", "Add">;

class TPU_TG_MemRef_FullyConnectedOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
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
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$filter,
    TPU_MemRefOfOrNone<[I32, I16]>:$bias,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    OptionalAttr<I8Attr>:$rshift,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}
def TPU_TG_MemRef_INT8_FullyConnectedOp : TPU_TG_MemRef_FullyConnectedOp<"tg_memref_int8_fully_connected", "Int8">;

class TPU_TG_MemRef_InputOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TG " # opQuant # " input operator.";

  let description = [{
    Produces a tensor as input data.

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
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}
def TPU_TG_MemRef_INT8_InputOp : TPU_TG_MemRef_InputOp<"tg_memref_int8_input", "Int8">;

def TPU_TG_MemRef_LoadWeightOp : TPU_Op<"tg_memref_load_weight", [NoSideEffect]> {
  let summary = "load_weight operator";

  let description = [{
    This is a virtual operation, to stand for loading a weight tensor,
    with specific tensor shape and type.

    Inputs:
      `weight_file`     : the virtual memref returned by `load_file` op.

    Attributes:
      `storage`         : represent the data type of the weight when save to
                          weight .bin file, setting to `NONE` means the weight
                          tensor is not present in the final weight bin file.
      `name`            : name of the tensor, this is the name showed in the
                          weight npz file, used as the key to retrieve tensor
                          data from the npz file.
      `lowered`         : indicating the weight data has been processed for
                          TPU hardware, eg. packed, transposed, etc.
      `compressed`      : indicating the weight data has been compressed to
                          utilize hardware decompression.
      `offset`          : global address of the weight tensor, which is an
                          offset within `WEIGHT` memory region.
    Result:
      `tensor`          : result tensor of the specific weight.

    Interfaces or Traits:
      `NoSideEffect`
  }];

  let arguments = (
    ins AnyMemRef:$weight_file,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    DefaultValuedAttr<TPU_WeightAttr, "FP32">:$storage,
    OptionalAttr<StrAttr>:$name,
    DefaultValuedAttr<BoolAttr, "false">:$lowered,
    DefaultValuedAttr<BoolAttr, "false">:$compressed,
    OptionalAttr<NonNegativeI64Attr>:$offset
  );
}

class TPU_TG_MemRef_Pool2DOp<string mnemonic, string opQuant, string opType> : TPU_Op<mnemonic,
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
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I8Attr>:$m_i8,
    TPU_PoolParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}
def TPU_TG_MemRef_INT8_PoolAvg2DOp : TPU_TG_MemRef_Pool2DOp<"tg_memref_int8_pool_avg_2d", "Int8", "Avg">;
def TPU_TG_MemRef_INT8_PoolMax2DOp : TPU_TG_MemRef_Pool2DOp<"tg_memref_int8_pool_max_2d", "Int8", "Max">;

def TPU_TG_MemRef_QuantOp: TPU_Op<"tg_memref_quant",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "Quantization/Dequantization operator";

  let description = [{
    Perform quantization or dequantization among different QuantMode.

    - From `None` to `INT8`
      Quantize a activation tensor into int8, according to its threshold value.
      Q(x) = x * 128 / threshold, and saturate to (-128, 127) range

    - From `INT8` to `None`
      Dequantize a activation tensor from int8 back to float.
      x = Q(x) * threshold / 128
  }];

  let arguments = (
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    TPU_QuantModeAttr:$from,
    TPU_QuantModeAttr:$to,
    OptionalAttr<F32Attr>:$threshold,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}

def TPU_TG_MemRef_ReshapeOp: TPU_Op<"tg_memref_reshape",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : to return previous Op`s quant info
      No `TpuOpLowerInterface`  : no lower
      `TpuTGOpCodegenInterface` : to return previous Op`s gaddr
  }];

  let arguments = (
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}

#endif // TPU_TG_OPS_MemRef
