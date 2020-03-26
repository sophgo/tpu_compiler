//===-- TPUTLOps.td - TPU dialect operation definitions ----*- tablegen -*-===//
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

//===----------------------------------------------------------------------===//
// TPU TG MemRef op definitions.
//===----------------------------------------------------------------------===//

def TPU_TG_TensorToMemRefOp : TPU_Op<"tensor_to_memref",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType]> {
  let summary = "tensor to memref operation";
  let description = [{
    The "tensor_load" operation creates a tensor from a memref, making an
    independent copy of the element data. The result value is a tensor whose
    shape and element type match the memref operand.

    Produce a value of tensor<4x?xf32> type.
       %12 = tensor_load %10 : memref<4x?xf32, #layout, memspace0>
  }];

  let arguments = (ins AnyTensor);
  let results = (outs AnyMemRef);
  // TensorLoadOp is fully verified by traits.
  let verifier = ?;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *tensor", [{
      auto tensorType = tensor->getType().cast<TensorType>();
      auto resultType = MemRefType::get(tensorType.getShape(),
                                        tensorType.getElementType());
      result.addOperands(tensor);
      result.addTypes(resultType);

      //auto memrefType = memref->getType().cast<MemRefType>();
      //auto resultType = RankedTensorType::get(memrefType.getShape(),
      //                                        memrefType.getElementType());
      //result.addOperands(memref);
      //result.addTypes(resultType);
  }]>];

  let extraClassDeclaration = [{
    // The result is always a memref.
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }

    /// The result of a tensor_load is always a tensor.
    // TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  }];
}

def TPU_TG_MemRefToTensorOp : TPU_Op<"memref_to_tensor",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType]> {
  let summary = "tensor load operation";
  let description = [{
    The "tensor_load" operation creates a tensor from a memref, making an
    independent copy of the element data. The result value is a tensor whose
    shape and element type match the memref operand.

    Produce a value of tensor<4x?xf32> type.
       %12 = tensor_load %10 : memref<4x?xf32, #layout, memspace0>
  }];

  let arguments = (ins AnyMemRef);
  let results = (outs AnyTensor);
  // TensorLoadOp is fully verified by traits.
  let verifier = ?;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *memref", [{
      auto memrefType = memref->getType().cast<MemRefType>();
      auto resultType = RankedTensorType::get(memrefType.getShape(),
                                              memrefType.getElementType());
      result.addOperands(memref);
      result.addTypes(resultType);
  }]>];

  let extraClassDeclaration = [{
    /// The result of a tensor_load is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  }];
}

class TPU_TG_MemRef_ConcatOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
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
    ins Variadic<MemRefOf<[AnyFloat, AnyInteger]>>:$inputs,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    OptionalAttr<I8Attr>:$rshift,
    OptionalAttr<I32ArrayAttr>:$m_i8_inputs,
    OptionalAttr<I8Attr>:$m_i8_output,
    I32Attr:$axis,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}
def TPU_TG_MemRef_INT8_ConcatOp : TPU_TG_MemRef_ConcatOp<"tg_memref_int8_concat", "Int8">;
def TPU_TG_MemRef_BF16_ConcatOp : TPU_TG_MemRef_ConcatOp<"tg_memref_bf16_concat", "Bf16">;


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
def TPU_TG_MemRef_INT8_PT_Conv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_int8_pt_conv_2d", "Int8 Per-Tensor">;
def TPU_TG_MemRef_INT8_PC_Conv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_int8_pc_conv_2d", "Int8 Per-Channel">;
def TPU_TG_MemRef_BF16_Conv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_bf16_conv_2d", "Bf16">;
def TPU_TG_MemRef_INT8_PT_DeConv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_int8_pt_deconv_2d", "Int8 Per-Tensor">;
def TPU_TG_MemRef_INT8_PC_DeConv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_int8_pc_deconv_2d", "Int8 Per-Channel">;
def TPU_TG_MemRef_BF16_DeConv2DOp : TPU_TG_MemRef_ConvOp<"tg_memref_bf16_deconv_2d", "Bf16">;

class TPU_TG_MemRef_CropOp<string mnemonic, string opQuant> : TPU_Op<mnemonic,
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
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    OptionalAttr<I32ArrayAttr>:$crop_shape,
    OptionalAttr<I32ArrayAttr>:$crop_offset,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}
def TPU_TG_MemRef_INT8_CropOp : TPU_TG_MemRef_CropOp<"tg_memref_int8_crop", "Int8">;
def TPU_TG_MemRef_BF16_CropOp : TPU_TG_MemRef_CropOp<"tg_memref_bf16_crop", "Bf16">;


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
def TPU_TG_MemRef_BF16_EltwiseAddOp : TPU_TG_MemRef_EltwiseOp<"tg_memref_bf16_eltwise_add", "Bf16", "Add">;

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
def TPU_TG_MemRef_BF16_FullyConnectedOp : TPU_TG_MemRef_FullyConnectedOp<"tg_memref_bf16_fully_connected", "Bf16">;

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
def TPU_TG_MemRef_BF16_PoolAvg2DOp : TPU_TG_MemRef_Pool2DOp<"tg_memref_bf16_pool_avg_2d", "Bf16", "Avg">;
def TPU_TG_MemRef_BF16_PoolMax2DOp : TPU_TG_MemRef_Pool2DOp<"tg_memref_bf16_pool_max_2d", "Bf16", "Max">;

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
