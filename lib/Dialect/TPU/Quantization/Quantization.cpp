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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/TPU/CustomOpPlugin.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include <sstream>
#include <fstream>

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


static void insertQuantOp(Operation *op) {
  auto builder = OpBuilder(op);

  StringRef curr_quant = isa<ReturnOp>(op) ? "NONE" : getOpQuant(op);
  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    auto prev_op = op->getOperand(i)->getDefiningOp();
    StringRef prev_quant;
    if (!prev_op) {
      prev_quant = "NONE";
    } else if (isa<tpu::QuantOp>(prev_op)
                || isa<tpu::LoadWeightOp>(prev_op)
                || isa<tpu::NoneOp>(prev_op)) {
      continue;
    } else {
      prev_quant = getOpQuant(prev_op);
    }

    // insert quant if prev and curr have different quant mode
    if (curr_quant != prev_quant) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(builder.getNamedAttr("from",
          builder.getStringAttr(prev_quant)));
      attrs.push_back(builder.getNamedAttr("to",
          builder.getStringAttr(curr_quant)));
      float threshold = 0.0f;
      std::string name;
      int layer_id = -1;
      if (curr_quant == "INT8") {
        threshold = getOpThreshold(prev_op);
        name = getOpName(prev_op).str() + "_quant";
        layer_id = getOpLayerId(op);
      } else if (prev_quant == "INT8") {
        threshold = getOpThreshold(prev_op);
        name = getOpName(prev_op).str() + "_dequant";
        layer_id = getOpLayerId(prev_op);
      } else if (curr_quant == "BF16") {
        name = getOpName(prev_op).str() + "_quant";
        layer_id = getOpLayerId(op);
      } else if (prev_quant == "BF16") {
        name = getOpName(prev_op).str() + "_dequant";
        layer_id = getOpLayerId(prev_op);
      }
      // app recognizes _quant as network output
      //name = name + "_" + prev_quant.str() + "_" + curr_quant.str();
      // check if prev op has inserted quant/dequant op
      if (prev_op) {
        bool found = false;
        for (auto &use : prev_op->getResult(0)->getUses()) {
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

      attrs.push_back(builder.getNamedAttr("threshold",
          builder.getF32FloatAttr(threshold)));
      attrs.push_back(builder.getNamedAttr("name",
          builder.getStringAttr(name)));
      attrs.push_back(builder.getNamedAttr("layer_id",
          builder.getI32IntegerAttr(layer_id)));

      auto shape = op->getOperand(i)->getType().cast<TensorType>().getShape();
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
          ArrayRef<Value *>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});

      op->setOperand(i, quantOp.getResult());

      LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                  << prev_quant << " => " << curr_quant << "\n";);
    }
  }
}

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

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);
    auto lrnOp = cast<tpu::LrnOp>(op);
    std::string quant = lrnOp.getOpQuant();

    if (quant == "NONE") {
      return matchFailure();
    }

    auto sq_table_op = lrnOp.getOperand(1)->getDefiningOp();
    if (isa<tpu::NoneOp>(sq_table_op) == false) {
      return matchFailure();
    }

    auto lrnThreeOp = lrnOp.getOperand(3)->getDefiningOp();
    if (isa<tpu::NoneOp>(lrnThreeOp) == true) {
      return matchFailure();
    }
    auto lrnTwoOp = lrnThreeOp->getOperand(0)->getDefiningOp();
    auto lrnOneOp = lrnTwoOp->getOperand(0)->getDefiningOp();

    // remote operand 3, not use any more
    lrnOp.setOperand(3, lrnOp.getOperand(1));

    if (quant == "INT8") {
      const int NPU_NUM = 32;
      const int TABLE_H_INT8 = 16;
      const int TABLE_W_INT8 = 16;
      const int TABLE_HW_INT8 = (TABLE_H_INT8 * TABLE_W_INT8);
      const int TBL_SHAPE_INT8 = (TABLE_HW_INT8 * NPU_NUM);
      auto lrnPartOp = cast<tpu::LrnThreeOp>(lrnThreeOp);
      uint32_t local_size = lrnPartOp.local_size().getLimitedValue();
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
          op->getLoc(), type, ArrayRef<Value *>{wfV},
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
          op->getLoc(), type, ArrayRef<Value *>{wfV},
          ArrayRef<NamedAttribute>{attrs2});
      lrnOp.setOperand(2, power_weight_op);
    }

    // remove lrn one/two/three op
    rewriter.replaceOp(lrnThreeOp, {lrnOp});
    rewriter.replaceOp(lrnTwoOp, {lrnOp});
    rewriter.replaceOp(lrnOneOp, {lrnOp});

    return matchSuccess();
  }
};

struct ExtendPreprocessOpPattern : public RewritePattern {
  ExtendPreprocessOpPattern(MLIRContext *context)
      : RewritePattern("tpu.preprocess", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto preprocessOp = cast<tpu::PreprocessOp>(op);

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

    // create int8 transpose
    int64_t tn, tc, th, tw;
    std::vector<NamedAttribute> transpose_attrs;
    std::string tranpose_name =
        getOpName(preprocessOp).str() + "_preprocess_tranpose";
    std::vector<int> transpose_orders;
    int layer_id = getOpLayerId(preprocessOp);
    if (preprocessOp.transpose_order().hasValue()) {
      for (auto m :
           llvm::enumerate(preprocessOp.transpose_order().getValue())) {
        auto attr = m.value().dyn_cast<IntegerAttr>();
        transpose_orders.push_back(attr.getInt());
      }
    }

    tn = input_shape[transpose_orders.at(0)];
    tc = input_shape[transpose_orders.at(1)];
    th = input_shape[transpose_orders.at(2)];
    tw = input_shape[transpose_orders.at(3)];

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
    transpose_attrs.push_back(
        builder.getNamedAttr("layer_id", builder.getI32IntegerAttr(layer_id)));

    auto transpose_type = RankedTensorType::get({tn, tc, th, tw}, eltType);
    auto transpose_op = OpBuilder(op).create<tpu::PermuteOp>(
        op->getLoc(), transpose_type, ArrayRef<Value *>{input_op},
        ArrayRef<NamedAttribute>{transpose_attrs});
    setOpThreshold(transpose_op, 127);
    setOpQuantParamType(transpose_op, "THRESHOLD");
    setOpQuant(transpose_op, "INT8");

    // create int8 crop
    std::string crop_name =
        getOpName(preprocessOp).str() + "_preprocess_crop";
    std::vector<int> crop_offset;
    if (preprocessOp.crop_offset().hasValue()) {
      for (auto m : llvm::enumerate(preprocessOp.crop_offset().getValue())) {
        auto attr = m.value().dyn_cast<IntegerAttr>();
        crop_offset.push_back(attr.getInt());
      }
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
    crop_attrs.push_back(
        builder.getNamedAttr("layer_id", builder.getI32IntegerAttr(layer_id)));
    // we only accept first input to IR, second input shape will be attribute.
    auto crop_op = OpBuilder(op).create<tpu::CropOp>(
        op->getLoc(), crop_type, ArrayRef<Value *>{transpose_op},
        ArrayRef<NamedAttribute>{crop_attrs});
    setOpThreshold(crop_op, 127);
    setOpQuantParamType(crop_op, "THRESHOLD");
    setOpQuant(crop_op, "INT8");

    // swapaxis, rgb to bgr or bgr to rgb
    std::string swapaxis_name =
        getOpName(preprocessOp).str() + "_preprocess_swapaxis";
    std::vector<int> color_orders;
    if (preprocessOp.color_order().hasValue()) {
      for (auto o : llvm::enumerate(preprocessOp.color_order().getValue())) {
        auto attr = o.value().dyn_cast<IntegerAttr>();
        color_orders.push_back(attr.getInt());
      }
    }
    auto swapaxis_type = RankedTensorType::get({on, oc, oh, ow}, eltType);
    std::vector<NamedAttribute> swapaxis_attrs;

    swapaxis_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(swapaxis_name)));
    swapaxis_attrs.push_back(builder.getNamedAttr(
        "channel_order",
        builder.getI32ArrayAttr(ArrayRef<int32_t>({color_orders}))));
    swapaxis_attrs.push_back(
        builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
    swapaxis_attrs.push_back(
        builder.getNamedAttr("layer_id", builder.getI32IntegerAttr(layer_id)));
    // we only accept first input to IR, second input shape will be attribute.
    auto swapaxis_op = OpBuilder(op).create<tpu::SwapChannelOp>(
        op->getLoc(), swapaxis_type, ArrayRef<Value *>{crop_op},
        ArrayRef<NamedAttribute>{swapaxis_attrs});
    setOpThreshold(swapaxis_op, 127);
    setOpQuantParamType(swapaxis_op, "THRESHOLD");
    setOpQuant(swapaxis_op, "INT8");

    // create bf16 mul

    // create bf16 add

    rewriter.replaceOp(preprocessOp, {swapaxis_op.getResult()});
    return matchSuccess();
  }
};

class TpuQuantPass : public FunctionPass<TpuQuantPass> {

public:
  explicit TpuQuantPass() {}

  void runOnFunction() override {
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
          || isa<tpu::ROIPoolingOp>(op)) {
        // continue
      } else if (isa<tpu::CustomOp>(op) &&
                 !cast<tpu::CustomOp>(op).do_quant()) {
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

    // unzip preprocess op
    OwningRewritePatternList preprocess_patterns;
    preprocess_patterns.insert<ExtendPreprocessOpPattern>(context);
    applyPatternsGreedily(fn, preprocess_patterns);

    // do quant
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::InputOp>(op)
          || isa<tpu::PreprocessOp>(op)
          || isa<tpu::QuantOp>(op)
          || isa<tpu::ReshapeOp>(op)
          || isa<tpu::ROIPoolingOp>(op)) {
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
            setOpResultType(op, StandardTypes::Integer, 8);
          } else if (getOpQuant(op) == "BF16") {
            plugin->bf16Quant(operation_name.c_str(), param, &quant, prevThreshold);
            setOpResultType(op, StandardTypes::BF16);
          }
          std::vector<NamedAttribute> newParam, newQuant;
          convertOpParamToAttributes(builder, param, newParam);
          convertOpParamToAttributes(builder, quant, newQuant);
          castOp.setAttr("param", DictionaryAttr::get(newParam, context));
          castOp.setAttr("quant", DictionaryAttr::get(newQuant, context));
        }
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (getOpQuant(op) == "INT8") {
          auto ret = quantOp.quantizeInt8();
          assert(!failed(ret));
        } else if (getOpQuant(op) == "BF16") {
          auto ret = quantOp.quantizeBf16();
          assert(!failed(ret));
        } else {
          llvm::errs() << "assert:" << op->getName() << "\n";
          assert(false);
        }
      }
    });

    // To make ReshapeOp's result element type same as
    // operand's after quantization
    fn.walk([&](tpu::ReshapeOp op) {
      auto _op = op.getOperation();
      auto eltType = _op->getOperand(0)->getType().cast<TensorType>().getElementType();
      auto shape = _op->getResult(0)->getType().cast<TensorType>().getShape();
      auto type = RankedTensorType::get(shape, eltType);
      _op->getResult(0)->setType(type);
    });

    // insert QuantOp if quant types don't equal.
    fn.walk([&](Operation *op) {
      if ((op->getName().getDialect().str() != "tpu"
           && !isa<ReturnOp>(op))
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else {
        insertQuantOp(op);
      }
    });

    OwningRewritePatternList patterns;
    // gen special operations
    patterns.insert<
      TpuGenLrnTablePattern
    >(context);
    applyPatternsGreedily(fn, patterns);
  }
};


namespace {
struct TpuTpuQuantClipPassPattern : public RewritePattern {
  TpuTpuQuantClipPassPattern(MLIRContext *context)
      : RewritePattern("tpu.clip", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto builder = OpBuilder(op);

    if (auto clipOp = llvm::dyn_cast<tpu::ClipOp>(op)) {
      // check threshold_max/threshold_min has assigned
      auto threshold_max = clipOp.quant().threshold_max().getValue().convertToFloat();
      auto threshold_min = clipOp.quant().threshold_min().getValue().convertToFloat();
      if (threshold_max == 0 && threshold_min == 0) {
        assert(0 && "you MUST do import-calibration-table before\n");
      }

      auto formerOp = clipOp.getOperand(0)->getDefiningOp();
      std::string formerOpName = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
      if (!formerOp->getResult(0)->hasOneUse()) {
        LLVM_DEBUG(llvm::errs() << "Not overwrtie more users op: " << formerOpName << ", not remove it\n";);
        return matchFailure();
      }

      auto layer_name = mlir::getOpName(clipOp).str();
      //bool in_black_list = std::find(clFuseClipLayers.begin(), clFuseClipLayers.end(), layer_name) != clFuseClipLayers.end();
      bool in_white_list = std::find(clSkipFuseClipLayers.begin(), clSkipFuseClipLayers.end(), layer_name) != clSkipFuseClipLayers.end();

      // white list priority is more than black one
      if (in_white_list) {
          LLVM_DEBUG(llvm::errs() << "config not quant op: " << layer_name << "\n";);
          return matchFailure();
      }

      if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
          LLVM_DEBUG(llvm::errs() << "over old " << mlir::getOpName(formerOp).str()
                  << " thre " << tpuOp.getOpQuantThreshold()
                  << ", new clip " << mlir::getOpName(clipOp).str()
                  << " thre is " << threshold_max << "\n";);
      }
      else {
        LLVM_DEBUG(llvm::errs() << "cant fuse previous op " << formerOpName << ", not remove it\n";);
        return matchFailure();
      }

      // update attr Only
      setOpThreshold(formerOp, threshold_max);
      formerOp->setAttr(llvm::StringRef("name"), rewriter.getStringAttr(layer_name));

      // remove clip
      rewriter.replaceOp(clipOp, {clipOp.getOperand(0)});

      return matchSuccess();
    }

    // default
    return matchFailure();
  }
};

class TpuQuantClipPass : public FunctionPass<TpuQuantClipPass> {
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
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTpuQuantPass() {
  return std::make_unique<TpuQuantPass>();
}

static PassRegistration<TpuQuantPass>
    pass("tpu-quant",
         "Do quantization on TPU Ops");

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTpuQuantClipPass() {
  return std::make_unique<TpuQuantClipPass>();
}

static PassRegistration<TpuQuantClipPass>
    pass_1("tpu-quant-clip",
         "merge clip's threshold to former");
