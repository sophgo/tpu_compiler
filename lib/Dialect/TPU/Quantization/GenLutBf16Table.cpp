//===- GenTanHTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include <float.h>

#define DEBUG_TYPE "gen-lut-table"

using namespace mlir;

// to cleanup
#if 0

struct TpuQuantTanHOpPattern : public RewritePattern {
  TpuQuantTanHOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tanh", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto tanhOp = cast<tpu::TanHOp>(op);
    std::string op_name = tanhOp.getAttrOfType<StringAttr>("name").getValue().str();
    //auto loc = op->getLoc();

    if (tanhOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << tanhOp.name() << " quantized already\n";);
      return matchFailure();
    }
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // find filter and bias tensor
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    // 0 is input
    int weight_idx = 0;
    for (unsigned i = 1; i < tanhOp.getNumOperands(); ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          tanhOp.getOperand(i)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      weights[weight_idx] = wTF->readTensor<float>(tensor_name, type);
      weight_idx++;
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    float *y0_table = (float *)weights[0]->data();
    float *scale = (float *)weights[1]->data();

    // create new tensors for quantized y0_table and scale
    auto y0_table_type = tanhOp.y0_table()->getType().cast<TensorType>();
    std::vector<int64_t> y0_table_shape(y0_table_type.getShape());
    assert(y0_table_shape.size() == 4);

    int64_t y0_table_size = std::accumulate(std::begin(y0_table_shape),
        std::end(y0_table_shape), 1, std::multiplies<>());

    // y0 / scale are same shape
    assert(y0_table_size == (int64_t)weights[0]->size());
    assert(y0_table_size == (int64_t)weights[1]->size());

    std::vector<bfloat16> new_y0_table(y0_table_size);
    std::vector<bfloat16> new_scale(y0_table_size);

    // quantization
    FloatToBFloat16(y0_table, new_y0_table.data(), y0_table_size);
    FloatToBFloat16(scale, new_scale.data(), y0_table_size);

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(tanhOp.getOperand(0)); // <! 0 is input

    // add new filter and bias weight
    std::vector<std::vector<bfloat16> *> newWeights{ &new_y0_table, &new_scale };
    std::vector<std::vector<int64_t> > weightShapes{ y0_table_shape, y0_table_shape};
    // 2 means y0_table / scale
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_quant_bf16_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";);

      auto type = RankedTensorType::get(weightShapes[i],
              FloatType::getBF16(rewriter.getContext()));

      wTF->addTensor<uint16_t>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("BF16")));

      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace with the new tanh op
    auto origAttrs = tanhOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("BF16")));
    rewriter.replaceOpWithNewOp<tpu::TanHOp>(
        tanhOp, tanhOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }
};

#endif
