//===- ConvertUpsampleToDeconv.cpp - convert unsample to deconv -----------===//
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
// This file convert upsample op to deconv.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_pool_mask"

using namespace mlir;

namespace {
struct TpuPoolMaskOpPattern : public RewritePattern {
  TpuPoolMaskOpPattern(MLIRContext *context)
      : RewritePattern("tpu.pool_mask", 7, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto poolMaskOp = cast<tpu::PoolMaskOp>(op);
    auto scale = poolMaskOp.scale().getLimitedValue();

    assert(op->getNumOperands() == 2 && "operands num should be 2");
    LLVM_DEBUG(llvm::errs() << poolMaskOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wFV = getWeightFileValue(op);

    std::vector<Value *> operands;
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> output_shape;
    operands.push_back(op->getOperand(0));
    auto input = op->getOperand(0);
    auto input_type = input->getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    output_shape.push_back(input_shape[0]);
    output_shape.push_back(input_shape[1]);
    output_shape.push_back(input_shape[2] * scale);
    output_shape.push_back(input_shape[3] * scale);
    RankedTensorType output_type = RankedTensorType::get(output_shape,
                                               input_type.getElementType());

    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());

    // y = upsample(x0)
    auto layer_name = poolMaskOp.name().str();
    auto name = layer_name + "_" + "upsample0";
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("scale",
                      rewriter.getI32IntegerAttr(scale)));

    auto upsample_op_0 = rewriter.create<tpu::UpsampleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // z = -1 * y + 0
    std::vector<float> const_scale_0(output_shape[1], -1);
    std::vector<float> const_bias_0(output_shape[1], 0);
    std::vector<int64_t> const_shape;
    const_shape.push_back(1);
    const_shape.push_back(output_shape[1]);
    const_shape.push_back(1);
    const_shape.push_back(1);
    auto scale_value_0 = addWeightTensorAndCreateWeightOp<float>(upsample_op_0,
                                 "scale", const_scale_0, const_shape,
                                 "NONE", wTF, wFV);
    auto bias_value_0 = addWeightTensorAndCreateWeightOp<float>(upsample_op_0,
                                 "bias", const_bias_0, const_shape,
                                 "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "scale_0";
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(upsample_op_0);
    operands.push_back(scale_value_0);
    operands.push_back(bias_value_0);
    auto scale_op_0 = rewriter.create<tpu::ScaleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // w = z + x1
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "eltwise_add_0";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(op->getOperand(1));
    operands.push_back(scale_op_0);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    auto eltwise_op_0 = rewriter.create<tpu::EltwiseAddOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // m = 1 * w + 1
    const float MIN_FLOAT = 1.0e-6;
    std::vector<float> const_scale_1(output_shape[1], 1);
    std::vector<float> const_bias_1(output_shape[1], MIN_FLOAT);

    auto scale_value_1 = addWeightTensorAndCreateWeightOp<float>(eltwise_op_0,
                                 "scale", const_scale_1, const_shape,
                                 "NONE", wTF, wFV);

    auto bias_value_1 = addWeightTensorAndCreateWeightOp<float>(eltwise_op_0,
                                 "bias", const_bias_1, const_shape,
                                 "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "scale_1";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(eltwise_op_0);
    operands.push_back(scale_value_1);
    operands.push_back(bias_value_1);
    auto scale_op_1 = rewriter.create<tpu::ScaleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // r = relu(m)
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "relu0";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(scale_op_1);
    auto relu_op_0 = rewriter.create<tpu::ReluOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);


    // q = r * [4, 3, 2, 1]
    auto count = output_shape[0] * output_shape[1] *
                 output_shape[2] * output_shape[3];

    std::vector<float> const_data(count, 0);
    for (int i = 0; i < output_shape[0] * output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        for (int k = 0; k < output_shape[3]; k++) {
          auto offset = i * output_shape[2] * output_shape[3] +
                        j * output_shape[3] + k;
          if (j % 2 == 0) {
            if ( k % 2 == 0)
              const_data[offset] = 4;
            if ( k % 2 == 1)
              const_data[offset] = 3;
          } else {
            if ( k % 2 == 0)
              const_data[offset] = 2;
            if ( k % 2 == 1)
              const_data[offset] = 1;
          }
        }
      }
    }
    auto const_value = addWeightTensorAndCreateWeightOp<float>(relu_op_0,
                                 "const", const_data, output_shape,
                                 "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "eltwise_mul_0";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(relu_op_0);
    operands.push_back(const_value);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto eltwise_mul_op_0 = rewriter.create<tpu::EltwiseMulOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // a = pool_max(q)
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "pool_max_0";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr("param",
                    tpu::PoolParam::get(
                         rewriter.getI32IntegerAttr(scale),
                         rewriter.getI32IntegerAttr(scale),
                         rewriter.getI32IntegerAttr(0),
                         rewriter.getI32IntegerAttr(0),
                         rewriter.getI32IntegerAttr(0),
                         rewriter.getI32IntegerAttr(0),
                         rewriter.getI32IntegerAttr(scale),
                         rewriter.getI32IntegerAttr(scale),
                         rewriter.getBoolAttr(false),
                         rewriter.getBoolAttr(true),
                         rewriter.getContext())));
    operands.push_back(eltwise_mul_op_0);
    auto pool_max_op_0 = rewriter.create<tpu::PoolMax2DOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{input_type}, operands, attrs);
    // b = upsample(a)
    attrs.clear();
    operands.clear();
    operands.push_back(pool_max_op_0);
    name = layer_name + "_" + "upsample_1";
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("scale",
                      rewriter.getI32IntegerAttr(scale)));

    auto upsample_op_1 = rewriter.create<tpu::UpsampleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // c = -1 * b + 0
    std::vector<float> const_scale_2(output_shape[1], -1);
    std::vector<float> const_bias_2(output_shape[1], 0);
    auto scale_value_2 = addWeightTensorAndCreateWeightOp<float>(upsample_op_1,
                                 "scale", const_scale_2, const_shape,
                                 "NONE", wTF, wFV);
    auto bias_value_2 = addWeightTensorAndCreateWeightOp<float>(upsample_op_1,
                                 "bias", const_bias_2, const_shape,
                                 "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "scale_2";
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(upsample_op_1);
    operands.push_back(scale_value_2);
    operands.push_back(bias_value_2);
    auto scale_op_2 = rewriter.create<tpu::ScaleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // d = c + q
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "eltwise_add_1";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(scale_op_2);
    operands.push_back(eltwise_mul_op_0);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto eltwise_op_1 = rewriter.create<tpu::EltwiseAddOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // e = d * 1 + 1
    std::vector<float> const_scale_3(output_shape[1], 1);
    std::vector<float> const_bias_3(output_shape[1], MIN_FLOAT);

    auto scale_value_3 = addWeightTensorAndCreateWeightOp<float>(eltwise_op_1,
                                 "scale", const_scale_3, const_shape,
                                 "NONE", wTF, wFV);

    auto bias_value_3 = addWeightTensorAndCreateWeightOp<float>(eltwise_op_1,
                                           "bias", const_bias_3, const_shape,
                                           "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "scale3";
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(eltwise_op_1);
    operands.push_back(scale_value_3);
    operands.push_back(bias_value_3);
    auto scale_op_3 = rewriter.create<tpu::ScaleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // f = relu(e)
    attrs.clear();
    operands.clear();
    name = layer_name + "_" + "relu_1";
    attrs.push_back(rewriter.getNamedAttr("quant",
                             getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(scale_op_3);
    auto relu_op_1 = rewriter.create<tpu::ReluOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // c = 1000000 * b + 0
    std::vector<float> const_scale_4(output_shape[1], 1.0e+6);
    std::vector<float> const_bias_4(output_shape[1], 0);
    auto scale_value_4 = addWeightTensorAndCreateWeightOp<float>(relu_op_1,
                                 "scale", const_scale_4, const_shape,
                                 "NONE", wTF, wFV);
    auto bias_value_4 = addWeightTensorAndCreateWeightOp<float>(relu_op_1,
                                 "bias", const_bias_4, const_shape,
                                 "NONE", wTF, wFV);

    attrs.clear();
    operands.clear();
    name = layer_name;
    attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(name)));
    operands.push_back(relu_op_1);
    operands.push_back(scale_value_4);
    operands.push_back(bias_value_4);
    auto scale_op_4 = rewriter.create<tpu::ScaleOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);


    rewriter.replaceOp(op, {scale_op_4});
    return matchSuccess();
}
};

class ConvertPoolMaskPass : public FunctionPass<ConvertPoolMaskPass> {
public:
  explicit ConvertPoolMaskPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuPoolMaskOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::PoolMaskOp::getCanonicalizationPatterns(
         OwningRewritePatternList &results,
         MLIRContext *context) {
  results.insert<TpuPoolMaskOpPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertPoolMaskPass() {
  return std::make_unique<ConvertPoolMaskPass>();
}

static PassRegistration<ConvertPoolMaskPass>
    pass("convert-poolmask",
         "Convert pool_mask");

