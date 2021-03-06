//===- ConvertNormalize.cpp - convert normalize ---------------------------===//
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
// This file implements the conversion of normalizaion.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_normalize"

using namespace mlir;

namespace {

struct TpuDecomposeNormalizePattern : public RewritePattern {
  TpuDecomposeNormalizePattern(MLIRContext *context)
      : RewritePattern("tpu.normalize", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto normalizeOp = cast<tpu::NormalizeOp>(op);
    LLVM_DEBUG(llvm::errs() <<normalizeOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    auto loc = op->getLoc();
    mlir::Value input_var = normalizeOp.getOperand(0);

    // op_name
    auto nameAttr = normalizeOp->getAttrOfType<StringAttr>("name");
    std::string op_name = nameAttr.getValue().str();
    LLVM_DEBUG(llvm::errs() << "Normalize Op: " << op_name << "\n";);

    // parse param
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(normalizeOp.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);

    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        normalizeOp.getOperand(1).getDefiningOp());
    assert(weight_op && "weight should be exist");
    auto tensor_name = weight_op.name();
    std::unique_ptr<std::vector<float> >  scale;

    auto type = weight_op.getResult().getType().cast<TensorType>();

    scale = wTF->readTensor<float>(tensor_name, type);
    wTF->deleteTensor<float>(tensor_name);

    auto result_type = normalizeOp.getResult().getType();

    LLVM_DEBUG(llvm::errs() << "Normalize Op tensor_name : "
                            << tensor_name << "\n";);

    ///
    /// separate Normalize op to below 6 ops.
    /// Eltwise OP(power(x,2))-> Reduction(use conv now)-> Sqrt-> Div->Eltwise OP(prod) ->Scale(by channel scale)
    ///

    /// 1. Power OP
    auto NoneOp = rewriter.create<tpu::NoneOp>(loc,rewriter.getNoneType());
    std::vector<Value> operands;
    operands.push_back(input_var);
    operands.push_back(NoneOp.getResult()); // quant_table
    operands.push_back(NoneOp.getResult()); // quant_table
    std::vector<NamedAttribute> attrs_power;
    attrs_power.push_back(rewriter.getNamedAttr("coeff", rewriter.getF32FloatAttr(2.0)));
    attrs_power.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name + "_eltwise_prod_power")));
    attrs_power.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    auto power_op = rewriter.create<tpu::PowOp>(
        rewriter.getUnknownLoc(), result_type,
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs_power});
    auto power_result_var = power_op.getResult();

    /// 2. Reduction(using conv2D Op) OP
    std::vector<Value> operands_conv;
    operands_conv.push_back(power_result_var);
    // - blobs_[0] holds the filter weights
    // - blobs_[1] holds the biases (optional)
    auto filter_name_conv = op_name +"_conv_filter";

    std::vector<float> weight(c*c,1);
    mlir::Type elementType_;
    elementType_ = FloatType::getF32(rewriter.getContext());
    //construct conv parameter
    //use C*C*1*1 filter to keep shape as input

    auto filter_type = RankedTensorType::get({c, c, 1, 1},elementType_);

    std::vector<int64_t> stride(2,1), dilation(2,1);
    bool is_dw = false , with_bias = false;
    int64_t g = 1;

    wTF->addTensor<float>(filter_name_conv, weight.data(), filter_type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(filter_name_conv)));

    auto weight_tensor = rewriter.create<tpu::LoadWeightOp>(loc, filter_type,
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});

    operands_conv.push_back(weight_tensor);
    operands_conv.push_back(NoneOp.getResult());
    operands_conv.push_back(NoneOp.getResult());  // quant_scale
    operands_conv.push_back(NoneOp.getResult());  // quant_zeropoint
    operands_conv.push_back(NoneOp.getResult());  // quant_rshift
    operands_conv.push_back(NoneOp.getResult());  // quant_multiplier

    std::vector<NamedAttribute> attrs_conv;
    attrs_conv.push_back(rewriter.getNamedAttr("name",
                         rewriter.getStringAttr(op_name+"_conv")));
    attrs_conv.push_back(rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs_conv.push_back(rewriter.getNamedAttr("param",
    tpu::ConvParam::get(
        rewriter.getI32IntegerAttr(1),
        rewriter.getI32IntegerAttr(1),
        rewriter.getI32IntegerAttr(stride[0]),
        rewriter.getI32IntegerAttr(stride[1]),
        rewriter.getStringAttr("VALID"),
        rewriter.getI32IntegerAttr(dilation[0]),
        rewriter.getI32IntegerAttr(dilation[1]),
        rewriter.getI32IntegerAttr(0), // pd_t
        rewriter.getI32IntegerAttr(0), // pd_b
        rewriter.getI32IntegerAttr(0), // pd_l
        rewriter.getI32IntegerAttr(0), // pd_r
        rewriter.getI32IntegerAttr(g),
        rewriter.getBoolAttr(is_dw),
        rewriter.getBoolAttr(with_bias),
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
        rewriter.getI32IntegerAttr(0), //pad_value
        rewriter.getContext())));
    attrs_conv.push_back(rewriter.getNamedAttr("quant",
                         getDefaultQuantParam(rewriter)));
    auto convOp = rewriter.create<tpu::Conv2DOp>(
        loc, result_type,
        ArrayRef<Value>{operands_conv}, ArrayRef<NamedAttribute>{attrs_conv});
    auto conv_result_var = convOp.getResult();

    /// 3. Sqrt OP
    std::vector<Value> operands_sqrt;
    operands_sqrt.push_back(conv_result_var);
    operands_sqrt.push_back(NoneOp.getResult()); // quant_table
    operands_sqrt.push_back(NoneOp.getResult()); // quant_table

    std::vector<NamedAttribute> attrs_sqrt;
    attrs_sqrt.push_back(rewriter.getNamedAttr(
        "name", rewriter.getStringAttr(op_name + "_sqrt")));
    attrs_sqrt.push_back(
        rewriter.getNamedAttr("coeff", rewriter.getF32FloatAttr(0.5)));
    attrs_sqrt.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    auto sqrt_op = rewriter.create<tpu::PowOp>(
        loc, result_type,
        ArrayRef<Value>{operands_sqrt}, ArrayRef<NamedAttribute>{attrs_sqrt});
    auto sqrt_result_var = sqrt_op.getResult();

    /// 4. Reciprocal OP
    std::vector<Value> operands_reciprocal;
    operands_reciprocal.push_back(sqrt_result_var);
    operands_reciprocal.push_back(NoneOp.getResult()); // quant_table
    operands_reciprocal.push_back(NoneOp.getResult()); // quant_table
    std::vector<NamedAttribute> attrs_reciprocal;
    attrs_reciprocal.push_back(rewriter.getNamedAttr("coeff", rewriter.getF32FloatAttr(-1.0f)));
    attrs_reciprocal.push_back(rewriter.getNamedAttr("name",
                               rewriter.getStringAttr(op_name+"_reciprocal")));
    attrs_reciprocal.push_back(rewriter.getNamedAttr("quant",
                               getDefaultQuantParam(rewriter)));
    auto reciprocal_op = rewriter.create<tpu::PowOp>(
        loc, result_type,
        ArrayRef<Value>{operands_reciprocal},
        ArrayRef<NamedAttribute>{attrs_reciprocal});
    auto reciprocal_result_var = reciprocal_op.getResult();

    /// 5. Eltwise_Mul OP
    std::vector<Value> operands_eltwise_mul;
    operands_eltwise_mul.push_back(input_var);
    operands_eltwise_mul.push_back(reciprocal_result_var);
    operands_eltwise_mul.push_back(NoneOp.getResult());
    operands_eltwise_mul.push_back(NoneOp.getResult());
    operands_eltwise_mul.push_back(NoneOp.getResult());
    operands_eltwise_mul.push_back(NoneOp.getResult());

    std::vector<NamedAttribute> attrs_eltwise_mul;
    attrs_eltwise_mul.push_back(rewriter.getNamedAttr("name",
                             rewriter.getStringAttr(op_name+"_eltwise_add")));
    attrs_eltwise_mul.push_back(rewriter.getNamedAttr("quant",
                                getDefaultQuantParam(rewriter)));

    auto eltwise_op = rewriter.create<tpu::EltwiseMulOp>(
        loc, result_type,
        ArrayRef<Value>{operands_eltwise_mul},
        ArrayRef<NamedAttribute>{attrs_eltwise_mul});
    auto eltwise_result_var = eltwise_op.getResult();

    /// 6. Scale OP
    std::vector<Value> operands_scale;
    operands_scale.push_back(eltwise_result_var);

    auto scale_name = op_name+"_scale_weight";
    auto scale_type = RankedTensorType::get({c}, elementType_);

    wTF->addTensor(scale_name, scale->data(), scale_type);
    std::vector<NamedAttribute> scale_weight_attrs;

    scale_weight_attrs.push_back(rewriter.getNamedAttr("name",
                                 rewriter.getStringAttr(scale_name)));
    weight_tensor = rewriter.create<tpu::LoadWeightOp>(loc, scale_type,
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{scale_weight_attrs});
    operands_scale.push_back(weight_tensor);
    //no bias , set none
    operands_scale.push_back(NoneOp.getResult());

    std::vector<NamedAttribute> scale_attrs;
    scale_attrs.push_back(rewriter.getNamedAttr("name",
                          rewriter.getStringAttr(op_name + "_scale")));
    auto scale_op = rewriter.create<tpu::ScaleOp>(
        loc, result_type, ArrayRef<Value>{operands_scale},
        ArrayRef<NamedAttribute>{scale_attrs});

    ///
    /// finally, replace NormalizeOp
    ///
    rewriter.replaceOp(normalizeOp, {scale_op});

    return success();
  }
};

class DecomposeNormalizePass : public mlir::PassWrapper<DecomposeNormalizePass, FunctionPass> {
public:
  explicit DecomposeNormalizePass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuDecomposeNormalizePattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::NormalizeOp::getCanonicalizationPatterns(
                                           OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<TpuDecomposeNormalizePattern>(context);
}

std::unique_ptr<mlir::Pass> mlir::createDecomposeNormalizePass() {
  return std::make_unique<DecomposeNormalizePass>();
}

