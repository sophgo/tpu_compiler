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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/MachineInfo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tpuc/CustomOpPlugin.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include <sstream>
#include <fstream>
#include <bmkernel/bm1880v2/1880v2_fp_convert.h>
#include "tpuc/MachineInfo.h"
#include <regex>

#define DEBUG_TYPE "quantization"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("TPU quantization options");

static llvm::cl::opt<bool> clQuantInt8PerTensor(
    "quant-int8-per-tensor",
    llvm::cl::desc("Disable per channel for convolution quantization"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantInt8RshiftOnly(
    "quant-int8-rshift-only",
    llvm::cl::desc("Disable multipler for convolution quantization"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantBf16(
    "quant-full-bf16",
    llvm::cl::desc("Quant to bf16 for all TPU ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixTable(
    "quant-int8-mix-bf16-table",
    llvm::cl::desc("Enable bf16 mix-presion from a table specifying mode for each TPU Op"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixSigmoid(
    "quant-int8-mix-bf16-sigmoid",
    llvm::cl::desc("Enable bf16 mix-presion on sigmoid Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixBroadcastMul(
    "quant-int8-mix-bf16-broadcastmul",
    llvm::cl::desc("Enable bf16 mix-presion on BroadcastMul Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixEltwiseMul(
    "quant-int8-mix-bf16-eltwisemul",
    llvm::cl::desc("Enable bf16 mix-presion on EltwiseMul Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixSoftmax(
    "quant-bf16-softmax",
    llvm::cl::desc("Enable bf16 Softmax Ops"),
    llvm::cl::init(true),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::list<std::string> clQuantLayer(
    "quant-int8-mix-bf16-layers",
    llvm::cl::desc("Enable bf16 mix-presion on specify layer"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clQuantLayerByFile(
    "quant-int8-mix-bf16-layers-from-file",
    llvm::cl::desc("Enable bf16 mix-presion on specify layers by file"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::list<std::string> clFuseClipLayers(
    "fuse-clip-layers",
    llvm::cl::desc("fuse clips by name"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clFuseClipLayersByFile(
    "fuse-clip-layers-from-file",
    llvm::cl::desc("fuse clips from file"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::list<std::string> clSkipFuseClipLayers(
    "skip-fuse-clip-layers",
    llvm::cl::desc("skip fuse clips by name"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clSkipFuseClipLayersByFile(
    "skip-fuse-clip-layers-from-file",
    llvm::cl::desc("skip fuse clips from file"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clSetLutMinMaxByFile(
    "set-lut-min-max-from-file",
    llvm::cl::desc("Set bf16 lut min/max range from file"),
    llvm::cl::cat(clOptionsCategory));

static inline bool is_fix8b(const StringRef&quant) {
  return quant == "INT8" || quant == "UINT8";
}

static void insertQuantOp(Operation *op) {
  auto builder = OpBuilder(op);

  if (isa<tpu::ReshapeOp>(op)) {
    return;
  }

  StringRef curr_quant = isa<ReturnOp>(op) ? "NONE" : getOpQuant(op);
  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    auto prev_op = op->getOperand(i).getDefiningOp();
    StringRef prev_quant;
    if (!prev_op) {
      prev_quant = "NONE";
    } else {
      if (isa<tpu::QuantOp>(prev_op)
                || isa<tpu::LoadWeightOp>(prev_op)
                || isa<tpu::NoneOp>(prev_op)) {
        continue;
      } else if (isa<tpu::ReshapeOp>(prev_op)) {
        prev_op = prev_op->getOperand(0).getDefiningOp();
      }
      if (auto castOp = dyn_cast<tpu::QuadraticSumOp>(prev_op)) {
        if (castOp.high_precision()) {
          prev_quant = "NONE";
        } else {
          prev_quant = getOpQuant(prev_op);
        }
      } else {
        prev_quant = getOpQuant(prev_op);
      }
    }

    if (prev_quant == "INT8" && isa<tpu::Yuv420CscOp>(prev_op)) {
      prev_quant = "UINT8";
    }

    // insert quant if prev and curr have different quant mode
    if (curr_quant != prev_quant && !(is_fix8b(prev_quant) && is_fix8b(curr_quant))) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(builder.getNamedAttr("from",
          builder.getStringAttr(prev_quant)));
      attrs.push_back(builder.getNamedAttr("to",
          builder.getStringAttr(curr_quant)));
      float threshold = 0.0f;
      int zero_point =0;
      std::string name;
      if (is_fix8b(curr_quant)) {
        threshold = getOpThreshold(prev_op);
        zero_point = getOpZeroPoint(prev_op);
        name = getOpName(prev_op).str() + "_quant";
      } else if (is_fix8b(prev_quant)) {
        threshold = getOpThreshold(prev_op);
        zero_point = getOpZeroPoint(prev_op);
        auto fuse_op = prev_op->getResult(0).use_begin()->getOwner();
        if (isa<tpu::ReshapeOp>(fuse_op)) {
          name = getOpName(fuse_op).str() + "_dequant";
        } else {
          name = getOpName(prev_op).str() + "_dequant";
        }
      } else if (curr_quant == "BF16") {
        threshold = getOpThreshold(prev_op);
        zero_point = getOpZeroPoint(prev_op);
        name = getOpName(prev_op).str() + "_quant";
      } else if (prev_quant == "BF16") {
        auto fuse_op = prev_op->getResult(0).use_begin()->getOwner();
        if (isa<tpu::ReshapeOp>(fuse_op)) {
          name = getOpName(fuse_op).str() + "_dequant";
        } else {
          name = getOpName(prev_op).str() + "_dequant";
        }
      }
      #if 1
      // app recognizes _quant as network output
      //name = name + "_" + prev_quant.str() + "_" + curr_quant.str();
      // check if prev op has inserted quant/dequant op
      if (prev_op) {
        bool found = false;
        for (auto &use : prev_op->getResult(0).getUses()) {
          auto nextOp = use.getOwner();
          if (getOpName(nextOp) == name) {
            op->setOperand(i, nextOp->getResult(0));
            LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                      << prev_quant << " => " << curr_quant << "\n";);
            found = true;
            break;
          }
        }
        if (found) {
          continue;
        }
      }
      #endif

      attrs.push_back(builder.getNamedAttr("threshold",
          builder.getF32FloatAttr(threshold)));
      attrs.push_back(builder.getNamedAttr("zero_point",
          builder.getI32IntegerAttr(zero_point)));
      attrs.push_back(builder.getNamedAttr("name",
          builder.getStringAttr(name)));

      auto shape = op->getOperand(i).getType().cast<TensorType>().getShape();
      Type eltType;
      if (curr_quant == "INT8") {
        eltType = IntegerType::get(8, builder.getContext());
      } else if (curr_quant == "BF16") {
        eltType = FloatType::getBF16(builder.getContext());
      } else {
        eltType = FloatType::getF32(builder.getContext());
      }
      auto type = RankedTensorType::get(shape, eltType);
      auto quantOp = builder.create<tpu::QuantOp>(op->getLoc(), type,
          ArrayRef<Value>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});

      op->setOperand(i, quantOp.getResult());

      LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                  << prev_quant << " => " << curr_quant <<  " threshold: "<< threshold<< " zero_point: " << zero_point << "\n";);
    }
  }
}

struct TpuConvertSoftmaxToSoftmaxCpu : public RewritePattern {
  TpuConvertSoftmaxToSoftmaxCpu(MLIRContext *context)
      : RewritePattern("tpu.softmax", 1, context) {}
      LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if(!clQuantMixSoftmax){
      std::vector<Value> operands;
      const int nInputs =  1;
      for (auto i = 0; i < nInputs; ++i) {
        operands.push_back(op->getOperand(i));
      }

        // Return same opValue
      auto loc = op->getLoc();
      auto newOp = rewriter.create<tpu::SoftmaxCpuOp>(loc,
        op->getResult(0).getType(),
        operands,
        op->getAttrs());

      // replace to relu->clip
      rewriter.replaceOp(op, {newOp});
      return success();
    }

    return failure();
  }
};

struct TpuGenLrnTablePattern : public RewritePattern {
  TpuGenLrnTablePattern(MLIRContext *context)
      : RewritePattern("tpu.lrn", 1, context) {}

  static void quantize_fraction(float x, float y, int &rshift_width,
                                int &x_quantized) {
    float y_ceiling = 256.0 / x * y;
    rshift_width = 0;
    x_quantized = 0;
    float y_quantized = 1.0;
    while ((y_quantized * 2) < y_ceiling) {
      rshift_width += 1;
      y_quantized = (float)(1 << rshift_width);
    }
    x_quantized = (int)std::floor((x / y) * y_quantized + 0.5);
  }

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    auto lrnOp = cast<tpu::LrnOp>(op);
    auto quant = lrnOp.getOpQuant();

    if (quant == "NONE") {
      return failure();
    }

    auto sq_table_op = lrnOp.getOperand(1).getDefiningOp();
    if (isa<tpu::NoneOp>(sq_table_op) == false) {
      return failure();
    }

    auto lrnThreeOp = lrnOp.getOperand(3).getDefiningOp();
    if (isa<tpu::NoneOp>(lrnThreeOp) == true) {
      return failure();
    }
    auto lrnTwoOp = lrnThreeOp->getOperand(0).getDefiningOp();
    auto lrnOneOp = lrnTwoOp->getOperand(0).getDefiningOp();

    // remote operand 3, not use any more
    lrnOp.setOperand(3, lrnOp.getOperand(1));

    const int EXP_START = -62;
    const int NPU_NUM = MInfo::lane_num;
    const int TABLE_H_INT8 = 16;
    const int TABLE_W_INT8 = 16;
    const int TABLE_HW_INT8 = (TABLE_H_INT8 * TABLE_W_INT8);
    const int TBL_SHAPE_INT8 = (TABLE_HW_INT8 * NPU_NUM);
    const int TABLE_H_BF16 = 32;
    const int TABLE_W_BF16 = 8;
    const int TABLE_HW_BF16 = ( TABLE_H_BF16 * TABLE_W_BF16);
    const int TBL_SHAPE_BF16 = ( TABLE_HW_BF16 * NPU_NUM );
    if (quant == "INT8") {
      auto lrnPartOp = cast<tpu::LrnThreeOp>(lrnThreeOp);
      uint32_t local_size = lrnPartOp.local_size();
      float alpha = lrnPartOp.alpha().convertToFloat();
      float beta = lrnPartOp.beta().convertToFloat();
      float k = lrnPartOp.k().convertToFloat();

      float sq_thy = getOpThreshold(lrnOneOp);
      float sumsq_thy = getOpThreshold(lrnTwoOp);
      float scale_thy = getOpThreshold(lrnThreeOp);
      float threshold_x = getPreviousOpThreshold(op);
      float threshold_y = getOpThreshold(op);
      // quant x and rshift
      int quant_x0, sum_rshift, quant_x1, lrn_rshift;
      quantize_fraction(sq_thy, sumsq_thy, sum_rshift, quant_x0);
      quantize_fraction(threshold_x * scale_thy, threshold_y * 256.0,
                        lrn_rshift, quant_x1);
      lrnOp.setAttr("sum_rshift", rewriter.getI32IntegerAttr(sum_rshift));
      lrnOp.setAttr("quant_data0", rewriter.getI32IntegerAttr(quant_x0));
      lrnOp.setAttr("lrn_rshift", rewriter.getI32IntegerAttr(lrn_rshift));
      lrnOp.setAttr("quant_data1", rewriter.getI32IntegerAttr(quant_x1));
      // sq table
      std::vector<float> sq_table(TBL_SHAPE_INT8);

      for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
        float lut_input = threshold_x / 128.0 * idx;
        float lut_output = std::pow(lut_input, 2) * 256.0 / sq_thy;
        lut_output = lut_output * alpha / local_size;
        lut_output = std::floor(lut_output + 0.5);
        if (lut_output > 255.0) {
          lut_output = 255.0;
        }
        for (int n = 0; n < NPU_NUM; n++) {
          sq_table[n * TABLE_HW_INT8 + idx] = lut_output;
        }
      }

      // power table
      std::vector<float> power_table(TBL_SHAPE_INT8);

      for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
        float lut_input = (float)idx / (256.0 / sumsq_thy);
        float lut_output = std::pow(lut_input + k, -beta);
        lut_output = lut_output * (256.0 / scale_thy);
        lut_output = std::floor(lut_output + 0.5);
        if (lut_output > 255.0) {
          lut_output = 255.0;
        }
        for (int n = 0; n < NPU_NUM; n++) {
          power_table[n * TABLE_HW_INT8 + idx] = lut_output;
        }
      }

      // update op params
      std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_INT8, TABLE_W_INT8};
      auto type = RankedTensorType::get(
          weightShape, FloatType::getF32(rewriter.getContext()));
      std::string op_name =
          lrnOp.getAttrOfType<StringAttr>("name").getValue().str();

      // sq weight
      auto tensor_name = op_name + "_sq_gen_weight";

      wTF->addTensor<float>(tensor_name, sq_table.data(), type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(
          rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
      auto sq_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      lrnOp.setOperand(1, sq_weight_op);

      // power weight
      auto tensor_name2 = op_name + "_power_gen_weight";
      wTF->addTensor<float>(tensor_name2, power_table.data(), type);
      std::vector<NamedAttribute> attrs2;
      attrs2.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name2)));
      attrs2.push_back(
          rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
      auto power_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs2});
      lrnOp.setOperand(2, power_weight_op);
    } else if (quant == "BF16"){
      auto lrnPartOp = cast<tpu::LrnThreeOp>(lrnThreeOp);
      float beta = lrnPartOp.beta().convertToFloat();

      // power table
      std::vector<uint16_t> power_exp_table_bf16(TBL_SHAPE_BF16);
      std::vector<uint16_t> power_mantissa_table_bf16(TBL_SHAPE_BF16);
      std::vector<float> power_exp_table(TBL_SHAPE_BF16);
      std::vector<float> power_mantissa_table(TBL_SHAPE_BF16);

      // gen exp table
      bf16_gen_power_exp_table(power_exp_table_bf16.data(), beta,
                               EXP_START, TABLE_HW_BF16);
      // gen matissa table
      bf16_gen_power_mantissa_table(power_mantissa_table_bf16.data(), beta,
                                    TABLE_HW_BF16);

      // copy bf16 data to float table
      for (int i = 0; i < NPU_NUM; ++i){
        std::copy(power_exp_table_bf16.data(), power_exp_table_bf16.data() + TABLE_HW_BF16,
                  power_exp_table.data() + i * TABLE_HW_BF16);
        std::copy(power_mantissa_table_bf16.data(),
                  power_mantissa_table_bf16.data() + TABLE_HW_BF16,
                  power_mantissa_table.data() + i * TABLE_HW_BF16);
      }

      // update op params
      std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_BF16, TABLE_W_BF16};
      auto type = RankedTensorType::get(
          weightShape, FloatType::getF32(rewriter.getContext()));
      std::string op_name =
          lrnOp.getAttrOfType<StringAttr>("name").getValue().str();

      // power exp weight
      auto tensor_name = op_name + "_power_exp_weight";

      wTF->addTensor<float>(tensor_name, power_exp_table.data(), type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(
          rewriter.getNamedAttr("storage", rewriter.getStringAttr("BF16")));
      auto power_exp_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      lrnOp.setOperand(1, power_exp_op);

      // power mantissa weight
      auto tensor_name2 = op_name + "_power_mantissa_weight";
      wTF->addTensor<float>(tensor_name2, power_mantissa_table.data(), type);
      std::vector<NamedAttribute> attrs2;
      attrs2.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name2)));
      attrs2.push_back(
          rewriter.getNamedAttr("storage", rewriter.getStringAttr("BF16")));
      auto power_mantissa_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs2});
      lrnOp.setOperand(2, power_mantissa_op);
    }

    // remove lrn one/two/three op
    rewriter.replaceOp(lrnThreeOp, {lrnOp});
    rewriter.replaceOp(lrnTwoOp, {lrnOp});
    rewriter.replaceOp(lrnOneOp, {lrnOp});

    return success();
  }
};

struct ExtendPreprocessOpPattern : public RewritePattern {
  ExtendPreprocessOpPattern(MLIRContext *context)
      : RewritePattern("tpu.preprocess", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto preprocessOp = cast<tpu::PreprocessOp>(op);
    auto nextOp = getNextOp(preprocessOp);
    assert(nextOp);
    auto elementType = nextOp->getResult(0).getType().template
                       cast<TensorType>().getElementType();

    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    auto builder = OpBuilder(op);
    auto input_op = op->getOperand(0);
    auto result = op->getResult(0);
    std::vector<int64_t> output_shape;
    std::vector<int64_t> input_shape;
    int64_t in, ic, ih, iw, on, oc, oh, ow, output_size, input_size;
    getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
    getTensorShapeAndSize(result, output_shape, output_size);
    getNCHW(input_shape, in, ic, ih, iw);
    getNCHW(output_shape, on, oc, oh, ow);

    Type eltType = FloatType::getF32(builder.getContext());
    std::vector<int> color_orders;
    if (preprocessOp.color_order().hasValue()) {
      for (auto o : llvm::enumerate(preprocessOp.color_order().getValue())) {
        auto attr = o.value().dyn_cast<IntegerAttr>();
        color_orders.push_back(attr.getInt());
      }
    }
    mlir::Value current_op = input_op;
    int64_t tn = in, tc = ic, th = ih, tw = iw;
    // create yuv420_csc
    if (preprocessOp.pixel_format().str() == "YUV420") {
      std::string name =
          getOpName(preprocessOp).str() + "_preprocess_yuv420_csc";
      auto yuv420_type =
          RankedTensorType::get({tn, tc, th, tw}, eltType);
      std::vector<NamedAttribute> attrs;

      attrs.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(name)));
      if (color_orders.empty() == false) {
        attrs.push_back(builder.getNamedAttr(
            "channel_order",
            builder.getI32ArrayAttr(ArrayRef<int32_t>({color_orders}))));
      }
      attrs.push_back(
          builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
      // we only accept first input to IR, second input shape will be attribute.
      auto yuv420_op = OpBuilder(op).create<tpu::Yuv420CscOp>(
          op->getLoc(), yuv420_type, ArrayRef<Value>{current_op},
          ArrayRef<NamedAttribute>{attrs});
      setOpThreshold(yuv420_op, 128);
      setOpQuantParamType(yuv420_op, "THRESHOLD");
      setOpQuant(yuv420_op, "UINT8");
      color_orders.clear();
      current_op = yuv420_op;
    }

    // create int8 transpose
    if (preprocessOp.transpose_order().hasValue()) {
      std::vector<NamedAttribute> transpose_attrs;
      std::string tranpose_name =
          getOpName(preprocessOp).str() + "_preprocess_tranpose";
      std::vector<int> transpose_orders;

      for (auto m :
           llvm::enumerate(preprocessOp.transpose_order().getValue())) {
        auto attr = m.value().dyn_cast<IntegerAttr>();
        transpose_orders.push_back(attr.getInt());
      }
      int64_t shape[4] = {tn,tc,th,tw};

      tn = shape[transpose_orders.at(0)];
      tc = shape[transpose_orders.at(1)];
      th = shape[transpose_orders.at(2)];
      tw = shape[transpose_orders.at(3)];

      transpose_attrs.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(tranpose_name)));
      transpose_attrs.push_back(builder.getNamedAttr(
          "order0", builder.getI32IntegerAttr(transpose_orders.at(0))));
      transpose_attrs.push_back(
          builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
      transpose_attrs.push_back(builder.getNamedAttr(
          "order1", builder.getI32IntegerAttr(transpose_orders.at(1))));
      transpose_attrs.push_back(builder.getNamedAttr(
          "order2", builder.getI32IntegerAttr(transpose_orders.at(2))));
      transpose_attrs.push_back(builder.getNamedAttr(
          "order3", builder.getI32IntegerAttr(transpose_orders.at(3))));

      auto transpose_type = RankedTensorType::get({tn, tc, th, tw}, eltType);
      auto transpose_op = OpBuilder(op).create<tpu::PermuteOp>(
          op->getLoc(), transpose_type, ArrayRef<Value>{current_op},
          ArrayRef<NamedAttribute>{transpose_attrs});
      setOpThreshold(transpose_op, 128);
      setOpQuantParamType(transpose_op, "THRESHOLD");
      setOpQuant(transpose_op, "UINT8");
      current_op = transpose_op;
    }

    // create uint8 pad
    if (preprocessOp.pads().hasValue()) {
      std::vector<NamedAttribute> pad_attrs;
      std::string pad_name = getOpName(preprocessOp).str() + "_preprocess_pad";
      std::vector<int> pads;
      bool no_pad = true;
      for (auto m : llvm::enumerate(preprocessOp.pads().getValue())) {
        int pad = m.value().dyn_cast<IntegerAttr>().getInt();
        pads.push_back(pad);
        if (pad != 0 && no_pad) {
          no_pad = false;
        }
      }
      if (no_pad)
        goto pad_exit;
      float const_val = preprocessOp.const_val().convertToFloat();

      tn = pads[0] + pads[4] + tn;
      tc = pads[1] + pads[5] + tc;
      th = pads[2] + pads[6] + th;
      tw = pads[3] + pads[7] + tw;

      pad_attrs.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(pad_name)));
      pad_attrs.push_back(builder.getNamedAttr(
          "pads", builder.getI32ArrayAttr(ArrayRef<int32_t>({pads}))));
      pad_attrs.push_back(builder.getNamedAttr(
          "const_val", builder.getF32FloatAttr(const_val)));
      pad_attrs.push_back(
          builder.getNamedAttr("quant", getDefaultQuantParam(builder)));

      auto pad_type = RankedTensorType::get({tn, tc, th, tw}, eltType);
      auto pad_op = OpBuilder(op).create<tpu::PadOp>(
          op->getLoc(), pad_type, ArrayRef<Value>{current_op},
          ArrayRef<NamedAttribute>{pad_attrs});
      setOpThreshold(pad_op, 128);
      setOpQuantParamType(pad_op, "THRESHOLD");
      setOpQuant(pad_op, "UINT8");
      current_op = pad_op;
    }
pad_exit:

    // create int8 crop
    if (preprocessOp.crop_offset().hasValue() && !(tn == on && tc == oc && th == oh && tw == ow)) {
      std::string crop_name =
          getOpName(preprocessOp).str() + "_preprocess_crop";
      std::vector<int> crop_offset;

      for (auto m : llvm::enumerate(preprocessOp.crop_offset().getValue())) {
        auto attr = m.value().dyn_cast<IntegerAttr>();
        crop_offset.push_back(attr.getInt());
      }

      std::vector<int> crop_shape(4);
      crop_shape.assign(output_shape.begin(), output_shape.end());
      auto crop_type = RankedTensorType::get({on, oc, oh, ow}, eltType);
      std::vector<NamedAttribute> crop_attrs;
      crop_attrs.push_back(builder.getNamedAttr(
          "crop_shape",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({crop_shape}))));
      crop_attrs.push_back(builder.getNamedAttr(
          "crop_offset",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({crop_offset}))));
      crop_attrs.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(crop_name)));
      crop_attrs.push_back(
          builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
      // we only accept first input to IR, second input shape will be attribute.
      auto crop_op = OpBuilder(op).create<tpu::CropOp>(
          op->getLoc(), crop_type, ArrayRef<Value>{current_op},
          ArrayRef<NamedAttribute>{crop_attrs});
      setOpQuantParamType(crop_op, "THRESHOLD");
      setOpThreshold(crop_op, 128);
      setOpQuant(crop_op, "UINT8");
      current_op = crop_op;
    }

    // create bf16 scale
    // ((x * raw_scale / 255.0) - mean / std) * scale
    // => x * scale / std - mean * scale /std
    std::string scale_name =
        getOpName(preprocessOp).str() + "_preprocess_scale";
    auto scale_type = RankedTensorType::get({on, oc, oh, ow}, eltType);
    float raw_scale = preprocessOp.raw_scale().convertToFloat();
    float scale = preprocessOp.scale().convertToFloat();
    std::vector<float> means;
    std::vector<float> stds;

    for (auto m : llvm::enumerate(preprocessOp.mean().getValue())) {
      auto attr = m.value().dyn_cast<FloatAttr>();
      means.push_back((float)attr.getValueAsDouble());
    }
    for (auto s : llvm::enumerate(preprocessOp.std().getValue())) {
      auto attr = s.value().dyn_cast<FloatAttr>();
      stds.push_back((float)attr.getValueAsDouble());
    }

    std::vector<float> scale_value(3), bias_value(3);
    auto scale_weight_type = RankedTensorType::get({oc, 1, 1, 1, 1}, eltType);
    auto bias_weight_type = RankedTensorType::get({oc}, eltType);
    std::vector<NamedAttribute> scale_weight_attrs, bias_weight_attrs;
    for (size_t i = 0; i < scale_value.size(); ++i) {
      scale_value[i] = (scale * raw_scale) / (stds[i] * 255.0);
      bias_value[i] = -(means[i] / stds[i]) * scale;
    }
    std::vector<float> tmp_scale_value(scale_value.begin(), scale_value.end());
    std::vector<float> tmp_bias_value(bias_value.begin(), bias_value.end());

    // swap op do after scale(because it can be fuesd with first conv)
    // change weight order here
    if (color_orders.size()){
      assert(color_orders.size() == 3 && "color_order must be 3");
      std::vector<float> tmp_scale_value(scale_value.begin(),
      scale_value.end());
      std::vector<float> tmp_bias_value(bias_value.begin(), bias_value.end());

      for(size_t i = 0; i < color_orders.size(); ++i){
        auto it = std::find(color_orders.begin(), color_orders.end(), i);
        int index = std::distance(color_orders.begin(), it);
        scale_value[i] = tmp_scale_value[index];
        bias_value[i] = tmp_bias_value[index];
      }
    }

    wTF->addTensor<float>(scale_name + "_0", scale_value.data(),
                          scale_weight_type);
    wTF->addTensor<float>(scale_name + "_1", bias_value.data(),
                          bias_weight_type);

    scale_weight_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(scale_name + "_0")));
    bias_weight_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(scale_name + "_1")));
    auto scale_weight_op = OpBuilder(op).create<tpu::LoadWeightOp>(
        op->getLoc(), scale_weight_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{scale_weight_attrs});
    auto bias_weight_op = OpBuilder(op).create<tpu::LoadWeightOp>(
        op->getLoc(), bias_weight_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{bias_weight_attrs});

    std::vector<Value> scale_operands;
    scale_operands.push_back(current_op);
    scale_operands.push_back(scale_weight_op);
    scale_operands.push_back(bias_weight_op);

    auto NoneOp = builder.create<tpu::NoneOp>(
        op->getLoc(), builder.getNoneType());
    scale_operands.push_back(NoneOp.getResult()); // quant_scale
    scale_operands.push_back(NoneOp.getResult()); // quant_zeropoint
    scale_operands.push_back(NoneOp.getResult()); // quant_rshift
    scale_operands.push_back(NoneOp.getResult());  // quant_multiplier

    std::vector<NamedAttribute> scale_attrs;
    scale_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(scale_name)));
    scale_attrs.push_back(builder.getNamedAttr("param",
        tpu::ConvParam::get(
            builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(1),
            builder.getStringAttr("VALID"),
            builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(1),
            builder.getI32IntegerAttr(0), // pd_t
            builder.getI32IntegerAttr(0), // pd_b
            builder.getI32IntegerAttr(0), // pd_l
            builder.getI32IntegerAttr(0), // pd_r
            builder.getI32IntegerAttr(oc),
            builder.getBoolAttr(true),
            builder.getBoolAttr(true),
            builder.getBoolAttr(false),
            builder.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
            builder.getI32IntegerAttr(0), //pad_value
            builder.getContext())));
    scale_attrs.push_back(
        builder.getNamedAttr("quant", getDefaultQuantParam(builder)));

    auto scale_op = OpBuilder(op).create<tpu::Conv2DOp>(
        op->getLoc(), scale_type, ArrayRef<Value>{scale_operands},
        ArrayRef<NamedAttribute>{scale_attrs});

    // to int8 as input, use input quantize threshold
    setOpThreshold(scale_op, getOpThreshold(op));
    setOpQuantParamType(scale_op, "THRESHOLD");
    setOpQuant(scale_op, "BF16");

    std::vector<int> no_change_swap = {0, 1, 2};

    if (std::equal(color_orders.begin(), color_orders.end(), no_change_swap.begin())) {
      rewriter.replaceOp(preprocessOp, {scale_op.getResult()});
      return success();
    }
    // swapaxis, rgb to bgr or bgr to rgb
    std::string swapaxis_name =
        getOpName(preprocessOp).str() + "_preprocess_swapaxis";

    auto swapaxis_type = RankedTensorType::get({on, oc, oh, ow}, eltType);
    std::vector<NamedAttribute> swapaxis_attrs;

    swapaxis_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(swapaxis_name)));
    swapaxis_attrs.push_back(builder.getNamedAttr(
        "channel_order",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({color_orders}))));
    swapaxis_attrs.push_back(
        builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
    // we only accept first input to IR, second input shape will be attribute.
    auto swapaxis_op = OpBuilder(op).create<tpu::SwapChannelOp>(
        op->getLoc(), swapaxis_type, ArrayRef<Value>{scale_op},
        ArrayRef<NamedAttribute>{swapaxis_attrs});
    setOpThreshold(swapaxis_op, getOpThreshold(op));
    setOpQuantParamType(swapaxis_op, "THRESHOLD");
    // the type of swapaxis_op should be bf16, if successor op is bf16.
    setOpQuant(swapaxis_op, elementType.isBF16() ? "BF16" : "INT8");

    rewriter.replaceOp(preprocessOp, {swapaxis_op.getResult()});
    return success();
  }
};

void setBF16LutMinMaxPattern(FuncOp& fn) {

  // parsing min / max ragne from file
  std::map<std::string, std::pair<float, float>> lutminmax_map;
  std::ifstream infile(clSetLutMinMaxByFile);
  std::string line;
  std::regex min_max_pattern("[a-zA-Z0-9.:_/-]+ [-0-9.e]+ [-0-9.e]+");
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string name;
    if (std::regex_match(line, min_max_pattern)) {
      float min, max;
      if (!(iss >> name >> min >> max)) { break; }
      LLVM_DEBUG(llvm::errs() << "  name " << name << ", min = "
                   << std::to_string(min) << ", max = "
                   << std::to_string(max) << "\n";);
      lutminmax_map[name] = std::make_pair(min, max);
    }
  }

  if (lutminmax_map.size()) {
    fn.walk([&](Operation *op) {
      if ((isa<tpu::ExpOp>(op)
            || isa<tpu::MishOp>(op)
            || isa<tpu::ReciprocalOp>(op)
            || isa<tpu::SigmoidOp>(op)
            || isa<tpu::SqrtOp>(op)
            || isa<tpu::ExpOp>(op)
            || isa<tpu::TanHOp>(op)
            ) && getOpQuant(op) == "BF16") {
        std::string op_name = mlir::getOpName(op).str();
        auto builder = OpBuilder(op);
        if (lutminmax_map.find(op_name) == lutminmax_map.end()) {
          LLVM_DEBUG(llvm::errs() << "not to find " << op_name << " in table\n";);
        }
        else {
          // symmetric / asymmetric case
          // symmetric: 0 symmetric such as -8 ~ 8
          // asymmetric: -16 ~ 0, add bias that set to symmetric and we could rewrite to -8 ~8
          std::pair<float, float> min_max = lutminmax_map[op_name];
          float min, max;
          std::tie(min, max) = min_max;
          float is_asymmetric = min + max;

          std::vector<NamedAttribute> attrs;
          attrs.push_back(builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
          //attrs.push_back(builder.getNamedAttr("layer_id", op->layer_idAttr()));

          auto loc = op->getLoc();
          auto NoneOp = builder.create<tpu::NoneOp>(loc, builder.getNoneType());
          Operation *_op = op;

          if (is_asymmetric) {
            // add bias to shift 0 as symmetric
            float zero_point = (max + min) / 2;
            float bias = -1.0 * zero_point;

            // add eltwise op, second input as bias, broadcast bias to hw
            // NOTICE: insert one is float32, quant as bf16
            // TODO: leverage add const
            std::vector<int64_t> shape;
            int64_t input_size;
            getTensorShapeAndSize(op->getOperand(0), shape, input_size);

            // add fp32 weight to npz
            std::unique_ptr<std::vector<float> >eltwise_second =
              std::make_unique<std::vector<float> >(input_size, bias);

            TensorFile *wTF = getWeightTensorFile(op);
            Value wfV = getWeightFileValue(op);
            StringRef storageType = "NONE";
            auto shuffix = "bias_zero_point_";
            auto name = op_name + "_" + shuffix + std::to_string(bias);
            auto weight_op = addWeightTensorAndCreateWeightOp<float>(
                op, shuffix, *eltwise_second, shape, storageType,
                wTF, wfV);

            attrs.push_back(builder.getNamedAttr("name",
                  builder.getStringAttr(name)));

            std::vector<Value> operands;
            operands.push_back(op->getOperand(0));
            operands.push_back(weight_op);

            operands.push_back(NoneOp.getResult());  // quant_scale
            operands.push_back(NoneOp.getResult());  // quant_zeropoint
            operands.push_back(NoneOp.getResult());  // quant_rshift
            operands.push_back(NoneOp.getResult());  // quant_multiplier

            auto eltwiseAddOp = builder.create<tpu::EltwiseAddOp>(
                loc, op->getResult(0).getType(),
                ArrayRef<Value>{operands},
                ArrayRef<NamedAttribute>{attrs});
            setOpQuant(eltwiseAddOp, "BF16");
            attrs.pop_back();

            // eltwise_add->lut->others
            op->setOperand(0, eltwiseAddOp.getResult());
            LLVM_DEBUG(llvm::errs() << "add zero_point : " << bias << ",";);
          }

          if (isa<tpu::MishOp>(op)) {
            // for high accuracy, we rewrite mish as x * tanh(softplus(x))
            // softplus
            auto name = op_name + "_softplus";
            attrs.push_back(builder.getNamedAttr("name",
                  builder.getStringAttr(name)));
            auto softplusOp = builder.create<tpu::SoftPlusOp>(
                loc, op->getResult(0).getType(),
                op->getOperands(),
                ArrayRef<NamedAttribute>{attrs});
            setOpQuant(softplusOp, "BF16");
            attrs.pop_back();

            // tanh
            std::vector<Value> operands;
            operands.push_back(softplusOp.getResult());
            operands.push_back(NoneOp.getResult());  // quant_scale
            operands.push_back(NoneOp.getResult());  // quant_zeropoint
            operands.push_back(NoneOp.getResult());  // quant_rshift
            operands.push_back(NoneOp.getResult());  // quant_multiplier

            name = op_name + "_tanh";
            attrs.push_back(builder.getNamedAttr("name",
                  builder.getStringAttr(name)));
            auto tanhOp = builder.create<tpu::TanHOp>(
                loc, softplusOp.getResult().getType(),
                operands,
                ArrayRef<NamedAttribute>{attrs});
            setOpQuant(tanhOp, "BF16");
            attrs.pop_back();
            _op = tanhOp;

            // eltwise
            operands.clear();
            operands.push_back(op->getOperand(0));
            operands.push_back(tanhOp.getResult());
            operands.push_back(NoneOp.getResult());  // quant_scale
            operands.push_back(NoneOp.getResult());  // quant_zeropoint
            operands.push_back(NoneOp.getResult());  // quant_rshift
            operands.push_back(NoneOp.getResult());  // quant_multiplier

            // collect all dependency before insert new relation
            SmallVector<Operation*, 4> uses;
            for (auto &use : op->getResult(0).getUses()) {
              // before: a->c
              // after : a->b->c
              Operation *owner = use.getOwner();
              uses.push_back(owner);
            }

            builder.setInsertionPointAfter(op);

            name = op_name;
            attrs.push_back(builder.getNamedAttr("name",
                  builder.getStringAttr(name)));

            auto eltwiseMulOp = builder.create<tpu::EltwiseMulOp>(
                loc, op->getResult(0).getType(),
                ArrayRef<Value>{operands},
                ArrayRef<NamedAttribute>{attrs});

            setOpQuant(eltwiseMulOp, "BF16");
            attrs.pop_back();

            // lut->mul(lut, x)->others
            for (auto &owner: uses) {
              owner->replaceUsesOfWith(op->getResult(0), eltwiseMulOp);
            }
          }

          LLVM_DEBUG(llvm::errs() << "is_symmetric: " << op_name << " min/max "
              << min << " / " << max << "\n";);
          _op->setAttr(llvm::StringRef("min_range"), builder.getF32FloatAttr(min));
          _op->setAttr(llvm::StringRef("max_range"), builder.getF32FloatAttr(max));
          _op->setAttr(llvm::StringRef("added_offset"), builder.getBoolAttr(true));
        }
      }
    });
  }
}

struct TpuTpuQuantClipPassPattern : public RewritePattern {
  TpuTpuQuantClipPassPattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto builder = OpBuilder(op);

    if (auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op)) {

      // check quant type
      auto formerOp = clipOp.getOperand(0).getDefiningOp();
      auto curr_quant = getOpQuant(op);
      auto prev_quant = getOpQuant(formerOp);
      auto next_quant = getOpQuant(op->getResult(0).getDefiningOp());

      // check threshold_max/threshold_min has assigned
      auto threshold_max = clipOp.quant().threshold_max().getValue().convertToFloat();
      auto threshold_min = clipOp.quant().threshold_min().getValue().convertToFloat();
      if (threshold_max == 0 && threshold_min == 0 && curr_quant == "INT8") {
        assert(0 && "you MUST do import-calibration-table before\n");
      }

      std::string formerOpName = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
      if (!formerOp->getResult(0).hasOneUse()) {
        LLVM_DEBUG(llvm::errs() << "Not overwrtie more users op: " << formerOpName << ", not remove it\n";);
        return failure();
      }

      auto layer_name = mlir::getOpName(clipOp).str();
      //bool in_black_list = std::find(clFuseClipLayers.begin(), clFuseClipLayers.end(), layer_name) != clFuseClipLayers.end();
      bool in_white_list = std::find(clSkipFuseClipLayers.begin(), clSkipFuseClipLayers.end(), layer_name) != clSkipFuseClipLayers.end();

      // white list priority is more than black one
      if (in_white_list) {
          LLVM_DEBUG(llvm::errs() << "config not quant op: " << layer_name << "\n";);
          return failure();
      }

      if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
          LLVM_DEBUG(llvm::errs() << "over old " << mlir::getOpName(formerOp).str()
                  << " thre " << tpuOp.getOpQuantThreshold()
                  << ", new clip " << mlir::getOpName(clipOp).str()
                  << " thre is " << threshold_max << "\n";);
      }
      else {
        LLVM_DEBUG(llvm::errs() << "cant fuse previous op " << formerOpName << ", not remove it\n";);
        return failure();
      }

      // always overwrite threshold for high accuracy
      if (curr_quant == "BF16" && prev_quant == "INT8" && next_quant == "INT8") {
        LLVM_DEBUG(llvm::errs() << "need to do in bf16 cuz prev/next is int8\n";);
        return failure();
      }

      if (curr_quant == "INT8" && prev_quant == "BF16") {
        // TODO: fuse relu to previous
        LLVM_DEBUG(llvm::errs() << "leave for relu\n";);
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(layer_name)));
        attrs.push_back(builder.getNamedAttr("quant", clipOp.quant()));

        auto op = rewriter.create<tpu::ReluOp>(
            clipOp.getLoc(), clipOp.getResult().getType(),
            ArrayRef<Value>{ clipOp.getOperand(0) },
            ArrayRef<NamedAttribute>{attrs});

        rewriter.replaceOp(clipOp, {op.getResult()});

        // overwrite previous one
        setOpThreshold(formerOp, threshold_max);
        return success();
      }

      if (prev_quant == "BF16") {
        LLVM_DEBUG(llvm::errs() << "no need to quant to int8 cuz former one " << formerOpName << " is bf16 quant type\n";);
        return failure();
      }

      // update attr Only
      setOpThreshold(formerOp, threshold_max);
      formerOp->setAttr(llvm::StringRef("name"), rewriter.getStringAttr(layer_name));

      // remove clip
      rewriter.replaceOp(clipOp, {clipOp.getOperand(0)});

      return success();
    }

    // default
    return failure();
  }
};


struct TpuConvertDilationWeightPattern : public RewritePattern {
  TpuConvertDilationWeightPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);

    auto dh = convOp.param().dilation_h().getInt();
    auto dw = convOp.param().dilation_w().getInt();
    const int DILATION_H_MAX = 15;
    const int DILATION_W_MAX = 15;
    if (dh <= DILATION_H_MAX && dw <= DILATION_W_MAX)
      return failure();

    TensorFile *wTF = getWeightTensorFile(op);
    auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
    std::vector<int64_t> filterShape;
    filterShape = getTensorShape(convOp.filter());

    int64_t oc = 0;
    int64_t ic = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    if (filterShape.size() == 4) {
      oc = filterShape[0];
      ic = filterShape[1];
      kh = filterShape[2];
      kw = filterShape[3];
    } else if (filterShape.size() == 5) {
      // g, oc/g, ic/g, kh, kw
      oc = filterShape[0] * filterShape[1];
      ic = filterShape[2];
      kh = filterShape[3];
      kw = filterShape[4];
    } else {
      assert(0);
    }

    int insertNumH = 0;
    int insertNumW = 0;
    int newDilationH = dh;
    int newDilationW = dw;
    while(1) {
      insertNumH++;
      newDilationH = (dh - 1 - insertNumH) / (insertNumH + 1) + 1;
      if (((dh - 1 - insertNumH) % (insertNumH + 1) == 0) &&
         newDilationH < DILATION_H_MAX)
        break;
    }

    while(1) {
      insertNumW++;
      newDilationW = (dw - 1 - insertNumW) / (insertNumW + 1) + 1;
      if (((dw - 1 - insertNumW) % (insertNumW + 1) == 0) &&
         newDilationW < DILATION_W_MAX)
        break;
    }

    int k_ext_h = (insertNumH + 1) * (kh - 1) + 1;
    int k_ext_w = (insertNumW + 1) * (kw - 1) + 1;
    filterShape[2] = k_ext_h;
    filterShape[3] = k_ext_w;
    auto filterSize = oc * ic * k_ext_h * k_ext_w;
    std::vector<float> newFilter(filterSize, 0);
    for (int i = 0; i < oc * ic; i++) {
      for (int j = 0; j < kh; j++) {
        for (int k = 0; k < kw; k++) {
          auto old_offset = i * kh * kw + j * kw + k;
          auto new_offset = i * k_ext_h * k_ext_w +
                            j * (insertNumW + 1) * k_ext_w +
                            k * (insertNumH + 1);
          newFilter[new_offset] = filter->data()[old_offset];
        }
      }
    }

    // update op
    if (getOpQuant(op) == "INT8")
      addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
          "dilation", newFilter, filterShape, "INT8", wTF);
    else if (getOpQuant(op) == "BF16")
      addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
          "dilation", newFilter, filterShape, "BF16", wTF);

    // rewrite pad
    convOp.setAttr("param",
           tpu::ConvParam::get(
                convOp.param().stride_h(),
                convOp.param().stride_w(),
                convOp.param().padding(),
                rewriter.getI32IntegerAttr(newDilationH),
                rewriter.getI32IntegerAttr(newDilationW),
                convOp.param().padding_t(),
                convOp.param().padding_b(),
                convOp.param().padding_l(),
                convOp.param().padding_r(),
                convOp.param().group(),
                convOp.param().is_dw(),
                convOp.param().with_bias(),
                convOp.param().do_relu(),
                convOp.param().ins(),
                convOp.param().pad_value(),
                rewriter.getContext()));

    return success();
  }
};

class TpuQuantPass : public mlir::PassWrapper<TpuQuantPass, FunctionPass> {

public:
  explicit TpuQuantPass() {}

  void runOnFunction() override {
    MInfo::getChipInfo(getFunction());
    assert(MInfo::version && "refer to set-chip");

    auto fn = getFunction();
    auto *context = &getContext();
    auto builder = Builder(context);

    // read mix precision from file, seperated by \n
    std::ifstream infile(clQuantLayerByFile);
    std::string line;
    while (std::getline(infile, line)) {
        clQuantLayer.push_back(line);
    }

    // mark quant mode
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::ReshapeOp>(op)
          || isa<tpu::InputOp>(op)
          || isa<tpu::PreprocessOp>(op)
          || isa<tpu::ROIPoolingOp>(op)
          || isa<tpu::SoftmaxCpuOp>(op)) {
        // continue
      } else if (isa<tpu::CustomOp>(op) &&
                 !cast<tpu::CustomOp>(op).do_quant()) {
        // continue
      } else if ((!clQuantMixSoftmax) && isa<tpu::SoftmaxOp>(op)) {
        // continue
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (clQuantMixTable) {
          //setOpQuant(op, quant_mix_table[getOpName(op)]);
          assert(false);
        } else if (!clQuantBf16) {
          setOpQuant(op, "INT8");
          if (isa<tpu::Conv2DOp>(op) || isa<tpu::DeConv2DOp>(op)) {
            if (clQuantInt8PerTensor) {
              setOpQuantPerchannel(op, false);
              setOpQuantParamType(op, "RSHIFT_ONLY");
            } else {
              setOpQuantPerchannel(op, true);
              if (clQuantInt8RshiftOnly) {
                setOpQuantParamType(op, "RSHIFT_ONLY");
              } else {
                setOpQuantParamType(op, "RSHIFT_AND_M_I32");
              }
            }
          }

          // mix-bf16 options
          if (clQuantMixSigmoid && isa<tpu::SigmoidOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (clQuantMixBroadcastMul && isa<tpu::BroadcastMulOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (clQuantMixEltwiseMul && isa<tpu::EltwiseMulOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (clQuantMixSoftmax && isa<tpu::SoftmaxOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (isa<tpu::SquareOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (isa<tpu::QuadraticSumOp>(op)) {
            setOpQuant(op, "BF16");
          }

          if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
            std::string layer_name = mlir::getOpName(op).str();
            if (std::find(clQuantLayer.begin(), clQuantLayer.end(), layer_name) != clQuantLayer.end()) {
              //llvm::errs() << "set " << layer_name << "as bf16\n";
              setOpQuant(op, "BF16");
            }
          }
        } else {
          setOpQuant(op, "BF16");
        }
      }
    });

    OwningRewritePatternList patterns;

    // check clip(relu6) is fused or leave for bf16
    // we implement relu6 with threshold, if no need quant(bf16 case)
    // we SHOULD do relu6 op
    patterns.insert<TpuTpuQuantClipPassPattern>(context);
    // patch for dialation > 15
    patterns.insert<TpuConvertDilationWeightPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // unzip preprocess op
    OwningRewritePatternList preprocess_patterns;
    preprocess_patterns.insert<ExtendPreprocessOpPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(preprocess_patterns));

    // set bf16 lut min/max range
    setBF16LutMinMaxPattern(fn);

    // do quant
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::InputOp>(op)
          || isa<tpu::PreprocessOp>(op)
          || isa<tpu::QuantOp>(op)
          || isa<tpu::ReshapeOp>(op)
          || isa<tpu::ROIPoolingOp>(op)
          || isa<tpu::SoftmaxCpuOp>(op)) {
      } else if (auto castOp = llvm::dyn_cast<tpu::CustomOp>(op)) {
        if (getOpQuant(op) != "NONE") {
          cvi::OpParam param, quant;
          auto operation_name = castOp.operation_name().str();
          float prevThreshold = getPreviousOpThreshold(op);
          convertAttributesToOpParam(castOp.param(), param);
          convertAttributesToOpParam(castOp.quant(), quant);
          cvi::CustomOpPlugin *plugin = cvi::CustomOpPlugin::load();
          assert(plugin);
          if (getOpQuant(op) == "INT8") {
            plugin->int8Quant(operation_name.c_str(), param, &quant, prevThreshold);
            setOpResultType(op->getResult(0), IntegerType::get(8, IntegerType::Signed, op->getContext()));
          } else if (getOpQuant(op) == "BF16") {
            plugin->bf16Quant(operation_name.c_str(), param, &quant, prevThreshold);
            setOpResultType(op->getResult(0), FloatType::getBF16(op->getContext()));
          }
          std::vector<NamedAttribute> newParam, newQuant;
          convertOpParamToAttributes(builder, param, newParam);
          convertOpParamToAttributes(builder, quant, newQuant);
          castOp.setAttr("param", DictionaryAttr::get(newParam, context));
          castOp.setAttr("quant", DictionaryAttr::get(newQuant, context));
        }
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (getOpQuant(op) == "INT8" || getOpQuant(op) == "UINT8") {
          auto ret = quantOp.quantizeInt8();
          assert(!failed(ret));
        } else if (getOpQuant(op) == "BF16") {
          auto ret = quantOp.quantizeBf16();
          assert(!failed(ret));
        } else if (isa<tpu::SoftmaxOp>(op)) {
          //do nothing
        } else {
          llvm::errs() << "assert:" << op->getName() << "\n";
          assert(false);
        }
      }
    });

    // insert QuantOp if quant types don't equal.
    fn.walk([&](Operation *op) {
      if ((op->getName().getDialect().str() != "tpu"
           && !isa<ReturnOp>(op))
          || isa<tpu::InputOp>(op)
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else {
        insertQuantOp(op);
      }
    });

    // To make ReshapeOp's result element type same as
    // operand's after quantization
    fn.walk([&](tpu::ReshapeOp op) {
      auto _op = op.getOperation();
      Operation* parantOp = _op->getOperand(0).getDefiningOp();
      auto name = parantOp->getName().getStringRef().str();
      Operation* inputQuantOp = NULL;

      if (name == "tpu.input") {
        // if reshape input is 'tpu.input', it should replace with it
        for (uint32_t i = 0; i < parantOp->getNumResults(); ++i) {
          for (auto &use : parantOp->getResult(i).getUses()) {
            Operation *owner = use.getOwner();
            name = owner->getName().getStringRef().str();
            if (name == "tpu.quant") {
              inputQuantOp = owner;
              // replace to quanted input
              _op->setOperand(0, inputQuantOp->getResult(0));
              break;
            }
          }
          if (inputQuantOp) {
            break;
          }
        }
      }

      auto eltType = _op->getOperand(0).getType().cast<TensorType>().getElementType();
      auto shape = _op->getResult(0).getType().cast<TensorType>().getShape();
      auto type = RankedTensorType::get(shape, eltType);
      _op->getResult(0).setType(type);
    });

    // gen special operations
    patterns.clear();
    patterns.insert<
      TpuGenLrnTablePattern
    >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
      TpuConvertSoftmaxToSoftmaxCpu
    >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};


namespace {
struct TpuTpuQuantClipPassPattern : public RewritePattern {
  TpuTpuQuantClipPassPattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op)) {
      // check threshold_max/threshold_min has assigned
      auto threshold_max = clipOp.quant().threshold_max().getValue().convertToFloat();
      auto threshold_min = clipOp.quant().threshold_min().getValue().convertToFloat();
      if (threshold_max == 0 && threshold_min == 0) {
        assert(0 && "you MUST do import-calibration-table before\n");
      }

      auto formerOp = clipOp.getOperand(0).getDefiningOp();
      std::string formerOpName = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
      if (!formerOp->getResult(0).hasOneUse()) {
        LLVM_DEBUG(llvm::errs() << "Not overwrtie more users op: " << formerOpName << ", not remove it\n";);
        return failure();
      }

      auto layer_name = mlir::getOpName(clipOp).str();
      //bool in_black_list = std::find(clFuseClipLayers.begin(), clFuseClipLayers.end(), layer_name) != clFuseClipLayers.end();
      bool in_white_list = std::find(clSkipFuseClipLayers.begin(), clSkipFuseClipLayers.end(), layer_name) != clSkipFuseClipLayers.end();

      // white list priority is more than black one
      if (in_white_list) {
          LLVM_DEBUG(llvm::errs() << "config not quant op: " << layer_name << "\n";);
          return failure();
      }

      if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
          LLVM_DEBUG(llvm::errs() << "over old " << mlir::getOpName(formerOp).str()
                  << " thre " << tpuOp.getOpQuantThreshold()
                  << ", new clip " << mlir::getOpName(clipOp).str()
                  << " thre is " << threshold_max << "\n";);
      }
      else {
        LLVM_DEBUG(llvm::errs() << "cant fuse previous op " << formerOpName << ", not remove it\n";);
        return failure();
      }

      // update attr Only
      setOpThreshold(formerOp, threshold_max);
      formerOp->setAttr(llvm::StringRef("name"), rewriter.getStringAttr(layer_name));

      // remove clip
      rewriter.replaceOp(clipOp, {clipOp.getOperand(0)});

      return success();
    }

    // default
    return failure();
  }
};

class TpuQuantClipPass : public mlir::PassWrapper<TpuQuantClipPass, FunctionPass> {
public:
  explicit TpuQuantClipPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {

    // black list
    std::ifstream infile(clFuseClipLayersByFile);
    std::string line;
    while (std::getline(infile, line)) {
        clFuseClipLayers.push_back(line);
    }

    // white list
    std::ifstream infile2(clSkipFuseClipLayersByFile);
    while (std::getline(infile2, line)) {
        clSkipFuseClipLayers.push_back(line);
    }

    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuTpuQuantClipPassPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createTpuQuantPass() {
  return std::make_unique<TpuQuantPass>();
}

std::unique_ptr<mlir::Pass> mlir::createTpuQuantClipPass() {
  return std::make_unique<TpuQuantClipPass>();
}
