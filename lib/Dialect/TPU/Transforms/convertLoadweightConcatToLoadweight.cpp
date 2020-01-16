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

namespace {

struct TpuConvertLoadeweightConcatToLoadweightPattern : public RewritePattern {
  TpuConvertLoadeweightConcatToLoadweightPattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.concat", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto concatOp = cast<tpu::ConcatOp>(op);

    int input_loadweight_num = concatOp.getOperands().size();

    /*
    There is 6 real input loadweight op for ssd , 
    plus one fake extral loadweight op to save result. 
    So need to input_loadweight_num-1 to do match check.
    */
    for(int i=0;i<input_loadweight_num-1;i++){
        auto formerOp = concatOp.getOperand(i)->getDefiningOp();
        if (!matchPattern(formerOp, m_Op<tpu::LoadWeightOp>())){
          return matchFailure();
        }
    }
    uint32_t  c, h, w;
    int tmp_w=0;
    llvm::errs() << "Starting to convert Layer " << concatOp.name().getValue() << "\n";
    auto result = concatOp.res();
    // LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    ///auto resultT = std::make_unique<std::vector<float> >(size);
    auto tmp_resultT = std::make_unique<std::vector<float> >(0);
    std::vector<float> resultT(size);

    std::vector<std::unique_ptr<std::vector<float> > > inputloadweight(input_loadweight_num);

    for (int i = 0; i < input_loadweight_num-1; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          concatOp.getOperand(i)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      inputloadweight[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }


    for (uint32_t i = 0; i < input_loadweight_num-1; i++) {
      std::vector<int64_t> shape =  concatOp.getOperand(i)->getType().cast<TensorType>().getShape();
      assert(3==shape.size()&&"only do 3 dim concat opt now");
      c = shape[0];
      h = shape[1];
      w = shape[2];

/*      llvm::errs() << "shape c:" << c <<"\n";
      llvm::errs() << "shape h:" << h << " w:"<< w <<"\n";*/

      float *input_data = (float *)inputloadweight[i]->data();

      for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
        auto shapeT = std::make_unique<std::vector<float> >(w);
        int insert_offset = ((idx_h+1)* tmp_w) + idx_h*w;
        shapeT.get()->assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
        tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
      }
      tmp_w += w;
    }

  resultT.assign(tmp_resultT.get()->begin(), tmp_resultT.get()->end());

  auto one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        concatOp.getOperand(input_loadweight_num-1)->getDefiningOp());
  auto weightFileVar = one_weight_op.getOperand();

  auto tensor_name = concatOp.getAttrOfType<StringAttr>("name").getValue().str() + "_loadweight" ;
  auto type = RankedTensorType::get(shape, FloatType::getF32(rewriter.getContext()));
  weightTensorFile_->addTensor<float>(tensor_name, &resultT, type);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
  auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
       ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});

  // replace concat with loadweight
  // the former one will be removed automatically

  rewriter.replaceOpWithNewOp<tpu::LoadWeightOp>(
      concatOp, new_weight_op.getResult()->getType(),
      ArrayRef<Value *>{weightFileVar},ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

class ConvertLoadeweightConcatToLoadweightPass : public FunctionPass<ConvertLoadeweightConcatToLoadweightPass> {
public:
  explicit ConvertLoadeweightConcatToLoadweightPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

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
    patterns.insert<TpuConvertLoadeweightConcatToLoadweightPattern>(context, weightTensorFile.get());
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

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertLoadeweightConcatToLoadweightPass() {
  return std::make_unique<ConvertLoadeweightConcatToLoadweightPass>();
}

static PassRegistration<ConvertLoadeweightConcatToLoadweightPass>
    pass("convert-loadweightconcat-to-loadweight",
         "convert loadweight bottom concat to loadweight");
