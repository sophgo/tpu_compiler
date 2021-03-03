//===- ConvertPriorBoxToLoadWeight.cpp - convert prior box to load weight -===//
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
// This file implements the conversion of prior box
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_priorbox"

using namespace mlir;

namespace {

struct TpuConvertLoadeweightConcatToLoadweightPattern : public RewritePattern {
  TpuConvertLoadeweightConcatToLoadweightPattern(MLIRContext *context)
      : RewritePattern("tpu.concat", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto concatOp = cast<tpu::ConcatOp>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    unsigned input_loadweight_num = concatOp.getOperands().size() - 4;

    for (int i = 0; i < (int)input_loadweight_num; i++) {
      auto formerOp = concatOp.getOperand(i).getDefiningOp();
      if (!isa<tpu::LoadWeightOp>(formerOp)) {
        return failure();
      }
    }
    LLVM_DEBUG(llvm::errs() << concatOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);

    uint32_t h, w;
    int tmp_w = 0;
    auto result = concatOp.getResult();
    // LLVM_DEBUG(llvm::errs() << "  result "; result.getType().dump();
    // llvm::errs() << "\n";);
    auto tensorType = result.getType().cast<TensorType>();
    std::vector<int64_t> shape = tensorType.getShape();
    auto resultT = std::make_unique<std::vector<float>>(0);

    std::vector<std::unique_ptr<std::vector<float>>> inputloadweight(
        input_loadweight_num);

    for (unsigned i = 0; i < input_loadweight_num; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          concatOp.getOperand(i).getDefiningOp());
      assert(weight_op && "weight op should be exist");
      auto tensor_name = weight_op.name();
      LLVM_DEBUG(llvm::errs()
                     << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult().getType().cast<TensorType>();
      inputloadweight[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    for (unsigned i = 0; i < input_loadweight_num; i++) {
      auto tensorType = concatOp.getOperand(i).getType().cast<TensorType>();
      std::vector<int64_t> shape = tensorType.getShape();
      assert(3 == shape.size() && "only do 3 dim concat opt now");
      h = shape[1];
      w = shape[2];

      // llvm::errs() << "shape h:" << h << " w:"<< w <<"\n";

      float *input_data = (float *)inputloadweight[i]->data();

      for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
        auto shapeT = std::make_unique<std::vector<float>>(w);
        int insert_offset = ((idx_h + 1) * tmp_w) + idx_h * w;
        shapeT.get()->assign(&input_data[idx_h * w],
                             &input_data[(idx_h + 1) * w]);
        resultT.get()->insert(resultT.get()->begin() + insert_offset,
                              shapeT->begin(), shapeT->end());
      }
      tmp_w += w;
    }

    auto tensor_name =
        concatOp->getAttrOfType<StringAttr>("name").getValue().str() +
        "_loadweight";
    auto type =
        RankedTensorType::get(shape, FloatType::getF32(rewriter.getContext()));
    wTF->addTensor<float>(tensor_name, resultT->data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(
        rewriter.getNamedAttr("storage", rewriter.getStringAttr("FP32")));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
        loc, type, ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});

    // replace concat with loadweight
    // the former one will be removed automatically

    rewriter.replaceOpWithNewOp<tpu::LoadWeightOp>(
        concatOp, new_weight_op.getResult().getType(), ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

struct TpuConvertPriorBoxPattern : public RewritePattern {
  TpuConvertPriorBoxPattern(MLIRContext *context)
      : RewritePattern("tpu.priorbox", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    auto loc = op->getLoc();
    auto priorboxOp = cast<tpu::PriorBoxOp>(op);
    auto result = priorboxOp.getResult();

    std::vector<int64_t> shape =
        result.getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float>>(size);
    std::vector<float> min_size;
    arrayAttrToVector(priorboxOp.min_size(), min_size);
    std::vector<float> max_size;
    arrayAttrToVector(priorboxOp.max_size(), max_size);
    std::vector<float> aspect_ratios;
    arrayAttrToVector(priorboxOp.aspect_ratios(), aspect_ratios);
    std::vector<float> variance;
    arrayAttrToVector(priorboxOp.variance(), variance);

    bool clip = priorboxOp.clip();
    bool use_default_aspect_ratio = priorboxOp.use_default_aspect_ratio();
    float offset = priorboxOp.offset().convertToFloat();
    float step_w = priorboxOp.step_w().convertToFloat();
    float step_h = priorboxOp.step_h().convertToFloat();
    int num_priors = priorboxOp.num_priors();
    int img_height = priorboxOp.img_h();
    int img_width = priorboxOp.img_w();

    if (max_size.size() > 0) {
      assert(max_size.size() == min_size.size() &&
             "num of max_size should the same with min_size");
    }
    // Must and only provide 4 variance.
    assert(variance.size() == 4 && "variance size must be 4");

    auto opd0Type = priorboxOp.getOperand(0).getType();
    auto opd1Type = priorboxOp.getOperand(1).getType();
    std::vector<int64_t> shape1 = opd1Type.cast<TensorType>().getShape();
    std::vector<int64_t> shape0 = opd0Type.cast<TensorType>().getShape();
    assert(shape1.size() == 4 && shape0.size() == 4);
    const int layer_width = shape0[3];
    const int layer_height = shape0[2];

    if (img_height == 0 || img_width == 0) {
      img_height = shape1[2];
      img_width = shape1[3];
    }

    if (step_w == 0 || step_h == 0) {
      step_w = static_cast<float>(img_width) / layer_width;
      step_h = static_cast<float>(img_height) / layer_height;
    }

    std::vector<float> top_data(size);

    int dim = layer_height * layer_width * num_priors * 4;
    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        float center_x = (w + offset) * step_w;
        float center_y = (h + offset) * step_h;
        float box_width, box_height;
        for (size_t s = 0; s < min_size.size(); ++s) {
          int min_size_ = (int)min_size[s];
          if (use_default_aspect_ratio) {
            // first prior: aspect_ratio = 1, size = min_size
            box_width = box_height = min_size_;
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }

          if (max_size.size() > 0) {
            int max_size_ = max_size[s];
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }

          // rest of priors
          for (size_t r = 0; r < aspect_ratios.size(); ++r) {
            float ar = aspect_ratios[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size_ * sqrt(ar);
            box_height = min_size_ / sqrt(ar);
            // xmin
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            // ymin
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            // xmax
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            // ymax
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
        }
      }
    }
    // clip the prior's coordidate such that it is within [0, 1]
    if (clip) {
      for (int d = 0; d < dim; ++d) {
        top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
      }
    }

    auto output_type = priorboxOp.output().getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());

    // set the variance.
    // top_data += (o_s[2]);

    int count = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
        for (int i = 0; i < num_priors; ++i) {
          for (auto &v : variance) {
            top_data[(o_s[2]) + count] = v;
            ++count;
          }
        }
      }
    }

    auto nameAttr = priorboxOp->getAttrOfType<StringAttr>("name");
    auto tensor_name = nameAttr.getValue().str() + "_loadweight";
    auto type =
        RankedTensorType::get(shape, FloatType::getF32(rewriter.getContext()));
    wTF->addTensor<float>(tensor_name, &top_data, type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
        loc, type, ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});

    // replace priorbox with loadweight
    // the former one will be removed automatically

    rewriter.replaceOpWithNewOp<tpu::LoadWeightOp>(
        priorboxOp, new_weight_op.getResult().getType(),
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

class ConvertPriorBoxPass : public mlir::PassWrapper<ConvertPriorBoxPass, FunctionPass> {
public:
  explicit ConvertPriorBoxPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertPriorBoxPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
    patterns.insert<TpuConvertLoadeweightConcatToLoadweightPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

// Canonicalizer
void tpu::PriorBoxOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuConvertPriorBoxPattern,
                 TpuConvertLoadeweightConcatToLoadweightPattern>(context);
}

std::unique_ptr<mlir::Pass> mlir::createConvertPriorBoxPass() {
  return std::make_unique<ConvertPriorBoxPass>();
}
