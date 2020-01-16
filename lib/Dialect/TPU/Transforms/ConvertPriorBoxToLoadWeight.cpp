//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace std;

namespace {

struct TpuConvertPriorBoxPattern : public RewritePattern {
  TpuConvertPriorBoxPattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.priorbox", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto priorboxOp = cast<tpu::PriorBoxOp>(op);
    auto result = priorboxOp.getResult();

    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    float min_size = priorboxOp.min_size().convertToFloat();
    float max_size = priorboxOp.max_size().convertToFloat();
    float aspect_ratio = priorboxOp.aspect_ratio0().convertToFloat();
    int aspect_ratios_size = priorboxOp.aspect_ratios_size().getLimitedValue();
    bool flip = priorboxOp.flip();
    bool clip = priorboxOp.clip();
    float variance0 = priorboxOp.variance0().convertToFloat();
    float variance1 = priorboxOp.variance1().convertToFloat();
    float variance2 = priorboxOp.variance2().convertToFloat();
    float variance3 = priorboxOp.variance3().convertToFloat();
    float offset = priorboxOp.offset().convertToFloat();
    float step = priorboxOp.step().convertToFloat();
    std::vector<float> min_sizes_;
    std::vector<float> max_sizes_;
    vector<float> aspect_ratios;
    vector<float> aspect_ratios_;
    bool flip_;
    int num_priors_;
    bool clip_;
    vector<float> variance_;
    int img_w_;
    int img_h_;
    float step_w_;
    float step_h_;

    float offset_;

    aspect_ratios.push_back(priorboxOp.aspect_ratio0().convertToFloat()) ;
    if(aspect_ratios_size==2)
      aspect_ratios.push_back(priorboxOp.aspect_ratio1().getValue().convertToFloat()) ;

    int max_size_size=priorboxOp.max_size_size().getLimitedValue();
    int min_size_size=priorboxOp.min_size_size().getLimitedValue();


  for (int i = 0; i < min_size_size; ++i) {
    min_sizes_.push_back(min_size);
    assert(min_sizes_.back()> 0 && "min_size must be positive.");
    assert(i==0); //more than one min size is not support. 
  }

    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.);
    flip_ = flip;
    for (int i = 0; i < aspect_ratios_size; ++i) {
          float ar = aspect_ratios[i];
          bool already_exist = false;
          for (int j = 0; j < aspect_ratios_.size(); ++j) {
            if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
              already_exist = true;
              break;
            }
          }
          if (!already_exist) {
            aspect_ratios_.push_back(ar);
            if (flip_) {
              aspect_ratios_.push_back(1./ar);
            }
          }
      }

    num_priors_ = aspect_ratios_.size() * min_sizes_.size();


    max_sizes_.push_back(max_size);
    assert(max_sizes_[0]> min_sizes_[0] && "max_size must be greater than min_size.");
    num_priors_ += 1;
      
    clip_ = clip;

    // Must and only provide 4 variance.
    assert(variance0> 0);
    variance_.push_back(variance0);
    assert(variance1> 0);
    variance_.push_back(variance1);    
    assert(variance2> 0);
    variance_.push_back(variance2);
    assert(variance3> 0);
    variance_.push_back(variance3);

    img_h_ = 0;
    img_w_ = 0;

    assert(step>0&&( "step should be larger than 0."));
    step_h_ = step;
    step_w_ = step;

    offset_ = offset;

  std::vector<int64_t> shape1 = priorboxOp.getOperand(1)->getType().cast<TensorType>().getShape();
  std::vector<int64_t> shape0 = priorboxOp.getOperand(0)->getType().cast<TensorType>().getShape();
  assert(shape1.size()==4&&shape0.size()==4);
  const int layer_width = shape0[3];
  const int layer_height = shape0[2];

  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = shape1[3];
    img_height = shape1[2];
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }


  std::vector<float> top_data(size);

  int dim = layer_height * layer_width * num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset_) * step_w;
      float center_y = (h + offset_) * step_h;
      float box_width, box_height;
      for (int s = 0; s < min_size_size; ++s) {
        int min_size_ = min_sizes_[s];
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

        if (max_size_size>0) {
          int max_size_ = max_sizes_[s];
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
        for (int r = 0; r < aspect_ratios_.size(); ++r) {
          float ar = aspect_ratios_[r];
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
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
    }
  }

  auto output_type = priorboxOp.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  // set the variance.
  //top_data += (o_s[2]);

  int count = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      for (int i = 0; i < num_priors_; ++i) {
        for (int j = 0; j < 4; ++j) {
          top_data[(o_s[2])+count] = variance_[j];
          ++count;
        }
      }
    }
  }
    auto one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          priorboxOp.getOperand(2)->getDefiningOp());
    auto weightFileVar = one_weight_op.getOperand();

  auto tensor_name = priorboxOp.getAttrOfType<StringAttr>("name").getValue().str() + "_loadweight" ;
  auto type = RankedTensorType::get(shape, FloatType::getF32(rewriter.getContext()));
  weightTensorFile_->addTensor<float>(tensor_name, &top_data, type);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
  auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
       ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});


  // replace priorbox with loadweight
  // the former one will be removed automatically

  rewriter.replaceOpWithNewOp<tpu::LoadWeightOp>(
      priorboxOp, new_weight_op.getResult()->getType(),
      ArrayRef<Value *>{weightFileVar},ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

class ConvertPriorBoxPass : public FunctionPass<ConvertPriorBoxPass> {
public:
  explicit ConvertPriorBoxPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    llvm::StringRef filename;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
    });
    auto weightTensorFile = openTensorFile(filename);

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertPriorBoxPattern>(context, weightTensorFile.get());
    applyPatternsGreedily(fn, patterns);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder b(fn.getBody());
      op.setAttr("filename", b.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertPriorBoxPass() {
  return std::make_unique<ConvertPriorBoxPass>();
}

static PassRegistration<ConvertPriorBoxPass>
    pass("convert-priorbox-to-loadweight",
         "convert priorbox to leadweight to save each priorbox result");
