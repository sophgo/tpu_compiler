//==================- PoolSplit.cpp
//------------------------------===//
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
//
//===----------------------------------------------------------------------===//


#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "split_pool"

namespace mlir{

namespace tpu {
struct SplitPoolPattern : public RewritePattern {
  SplitPoolPattern(MLIRContext *context, MInfo &mInfo)
      : RewritePattern("tpu.tg_int8_pool_avg_2d", 1, context), mInfo(mInfo) {
  }

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {



    auto avg_pool_op = cast<TG_INT8_PoolAvg2DOp>(op);
    if (!avg_pool_op) {
      return failure();
    }
    auto formerOp = op->getOperand(0);


    std::vector<int64_t> input_shape = getTensorShape(formerOp);
    int64_t n, c, ih, iw;
    getNCHW(input_shape, n, c, ih, iw);
    int64_t output_size = getTensorSize(avg_pool_op);
    uint64_t lmem_size = mInfo.lmem_per_lane;
    if ((uint64_t)(ih * iw) < ((lmem_size - output_size) / 2)) {
      return failure();
    }
    LLVM_DEBUG(llvm::errs() << "TG split avg pool");
    LLVM_DEBUG(llvm::errs() << "(input_h * input_w ) "<<ih * iw << " >> (local memory size)" << lmem_size << " output_size " << output_size << "\n";);

    std::vector<int> h_slices;
    int h_slice_size = (int)(((lmem_size - output_size)/iw)/2);
    int total_h = ih;
    while(total_h > 0){
      if(total_h > h_slice_size){
        total_h-=h_slice_size;
        h_slices.push_back(h_slice_size);
      }else{
        h_slices.push_back(total_h);
        break;
      }
    }

    std::vector<Value> all_slice_avg_op;

    int offset = 0;
    mlir::TensorType result_type;
    auto elementType_ = formerOp.getType().cast<TensorType>().getElementType();
    for (auto &slice : h_slices) {
      std::vector<NamedAttribute> attrs;
      result_type = RankedTensorType::get({n, c, slice, iw}, elementType_);
      std::vector<int> crop_offset = {0, 0, offset, 0};
      attrs.push_back(
          rewriter.getNamedAttr("crop_offset", rewriter.getI32ArrayAttr(crop_offset)));

      offset += slice;
      attrs.push_back(rewriter.getNamedAttr(
          "name", rewriter.getStringAttr("slice_" + getOpName(op).str() +
                                         std::to_string(offset))));

      auto splitOp = rewriter.create<tpu::TG_INT8_CropOp>(
          op->getLoc(), result_type, ArrayRef<Value>{{formerOp}},
          ArrayRef<NamedAttribute>{attrs});
      attrs.clear();

      result_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
      attrs.push_back(rewriter.getNamedAttr(
          "param",
          tpu::PoolParam::get(rewriter.getI32IntegerAttr(slice), // kernel_h
                              rewriter.getI32IntegerAttr(iw),    // kernel_w
                              rewriter.getI32IntegerAttr(0),     // padding_t
                              rewriter.getI32IntegerAttr(0),     // padding_b
                              rewriter.getI32IntegerAttr(0),     // padding_l
                              rewriter.getI32IntegerAttr(0),     // padding_r
                              rewriter.getI32IntegerAttr(0),     // pad_value
                              rewriter.getI32IntegerAttr(1),     // stride_h
                              rewriter.getI32IntegerAttr(1),     // stride_w
                              rewriter.getBoolAttr(false),
                              rewriter.getContext())));
      attrs.push_back(rewriter.getNamedAttr("m_i8", avg_pool_op.m_i8Attr()));
      attrs.push_back(rewriter.getNamedAttr("rshift", avg_pool_op.rshiftAttr()));
      attrs.push_back(rewriter.getNamedAttr(
          "name", rewriter.getStringAttr("pool_" + getOpName(op).str() +
                                         std::to_string(offset))));
      auto pool_op = rewriter.create<tpu::TG_INT8_PoolAvg2DOp>(
          op->getLoc(), result_type, ArrayRef<Value>{{splitOp}},
          ArrayRef<NamedAttribute>{attrs});
      all_slice_avg_op.push_back(pool_op);
    }
    std::vector<NamedAttribute> final_attrs;
    final_attrs.push_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    final_attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(2)));
    final_attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr("concat")));
    int h_slices_num = h_slices.size();
    std::vector<int32_t> m_i8_inputs_array(h_slices_num, 1);
    final_attrs.push_back(rewriter.getNamedAttr(
        "m_i8_inputs",
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    std::vector<int32_t> rshift_array(h_slices_num, 0);
    final_attrs.push_back(rewriter.getNamedAttr(
        "rshift", rewriter.getI32ArrayAttr(ArrayRef<int32_t>({rshift_array}))));

    result_type = RankedTensorType::get({n, c, h_slices_num, 1}, elementType_);
    auto concat_op = rewriter.create<tpu::TG_INT8_ConcatOp>(
        op->getLoc(), result_type, ArrayRef<Value>{all_slice_avg_op},
        ArrayRef<NamedAttribute>{final_attrs});
    final_attrs.clear();
    result_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
    final_attrs.push_back(
        rewriter.getNamedAttr("m_i8", rewriter.getI8IntegerAttr(1)));
    final_attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI8IntegerAttr(0)));
    final_attrs.push_back(rewriter.getNamedAttr(
        "param",
        tpu::PoolParam::get(
            rewriter.getI32IntegerAttr(h_slices_num), // kernel_h
            rewriter.getI32IntegerAttr(1),            // kernel_w
            rewriter.getI32IntegerAttr(0),            // padding_t
            rewriter.getI32IntegerAttr(0),            // padding_b
            rewriter.getI32IntegerAttr(0),            // padding_l
            rewriter.getI32IntegerAttr(0),            // padding_r
            rewriter.getI32IntegerAttr(0),            // pad_value
            rewriter.getI32IntegerAttr(1),            // stride_h
            rewriter.getI32IntegerAttr(1),            // stride_w
            rewriter.getBoolAttr(false), rewriter.getContext())));
    final_attrs.push_back(rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    final_attrs.push_back(rewriter.getNamedAttr("name", avg_pool_op.nameAttr()));
    auto pool_final_op = rewriter.create<tpu::TG_INT8_PoolAvg2DOp>(
        op->getLoc(), result_type, ArrayRef<Value>{{concat_op}},
        ArrayRef<NamedAttribute>{final_attrs});
    rewriter.replaceOp(op, {pool_final_op});

    return success();
  };
  MInfo &mInfo;
};

void PopulateSplitPoolPatterns(MLIRContext *context,
                               OwningRewritePatternList *patterns,
                               MInfo &mInfo) {
  patterns->insert<SplitPoolPattern>(context, mInfo);
}

} // namespace
}
