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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"

using namespace mlir;

namespace {

struct TpuNormalizePattern : public RewritePattern {
  TpuNormalizePattern(MLIRContext *context, TensorFile *weightTensorFile,Value* weightFileVar)
      : RewritePattern("tpu.normalize", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar){}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto normalizeOp = cast<tpu::NormalizeOp>(op);
    llvm::errs() <<normalizeOp.getOperationName() << "\n";
    auto loc = op->getLoc();
    mlir::Value *input_var = normalizeOp.getOperand(0);
 
    // op_name
    std::string op_name = normalizeOp.getAttrOfType<StringAttr>("name").getValue().str();
    llvm::errs() << "Normalize Op: " << op_name << "\n";

    // parse param
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(normalizeOp.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);

    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        normalizeOp.getOperand(1)->getDefiningOp());
    assert(weight_op);

    auto type = weight_op.getResult()->getType().cast<TensorType>();
    float *scale = (float*)weightTensorFile_->readTensor<float>(op_name, type)->data();
    weightTensorFile_->deleteTensor<float>(op_name);

    auto result_type = normalizeOp.getResult()->getType();

/*
  Currenly , we separate Normalize op to below 6 ops.
  Eltwise OP(power(x,2))-> Reduction(use conv now)-> Sqrt-> Div->Eltwise OP(prod) ->Scale(by channel scale)
*/

  /* 1. Power OP */
#if 0
  //use power op

  assert(false&&"not support normalize decompose to power");
  std::vector<Value *> operands;
  operands.push_back(input_var);

  // we leverage depthwise to calculat a*x + b,
  // one shot by channel, we should reserve weight
  // for extend scale/shift from 1 dimension to <1, NUP_NUM, 1, 1>
  // FIXME: not harcode
  int channel = 32;
  int tbl_size = channel;
  auto table_type_scale = RankedTensorType::get({1, channel, 1, 1}, elementType_);
  std::vector<float> dataVec_fp32;
  dataVec_fp32.reserve(tbl_size);
  auto filter_name = layer->layer_param().name()+"_power_scale";
  weightFile_->addTensor(filter_name, &dataVec_fp32, table_type_scale);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type_scale));

  // we just allocate 1 batch cuz
  // `AssignWeightAddress.cpp` auto seperate high/low part into int8 buffer
  tbl_size = 1 * channel;
  auto table_type_shift = RankedTensorType::get({1, channel, 1, 1}, elementType_);
  dataVec_fp32.reserve(tbl_size);
  filter_name = layer->layer_param().name()+"_power_shift";
  weightFile_->addTensor(filter_name, &dataVec_fp32, table_type_shift);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type_shift));

  auto result_type = RankedTensorType::get(input_shape, elementType_);
  std::vector<NamedAttribute> attrs_power;
  attrs_power.push_back(builder_.getNamedAttr("power", builder_.getF32FloatAttr(2.0)));
  attrs_power.push_back(builder_.getNamedAttr("scale", builder_.getF32FloatAttr(1.0)));
  attrs_power.push_back(builder_.getNamedAttr("shift", builder_.getF32FloatAttr(0.0)));
  attrs_power.push_back(builder_.getNamedAttr("rshift", builder_.getF32FloatAttr(0.0)));

  attrs_power.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name()+"_power")));

  auto power_op = rewriter.create<tpu::PowerOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs_power});
  auto power_result_var = power_op.getResult();
#else  
  /*use eltwise op*/
  std::vector<Value *> operands_eltwise_power;

  operands_eltwise_power.push_back(input_var);
  operands_eltwise_power.push_back(input_var);

  auto NoneOp = rewriter.create<tpu::NoneOp>(loc,rewriter.getNoneType());
  operands_eltwise_power.push_back(NoneOp.getResult());
  operands_eltwise_power.push_back(NoneOp.getResult());
  operands_eltwise_power.push_back(NoneOp.getResult());
  operands_eltwise_power.push_back(NoneOp.getResult());

  std::vector<NamedAttribute> attrs_eltwise_power;
  attrs_eltwise_power.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name+"_eltwise_prod_power")));
  attrs_eltwise_power.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

   auto eltwiseMulOp = rewriter.create<tpu::EltwiseMulOp>(
      loc, result_type,
      ArrayRef<Value *>{operands_eltwise_power}, ArrayRef<NamedAttribute>{attrs_eltwise_power});

  // rewriter.replaceOpWithNewOp<tpu::EltwiseMulOp>(
  //     normalizeOp, eltwiseMulOp.getResult()->getType(),
  //     ArrayRef<Value *>{operands_eltwise_power},ArrayRef<NamedAttribute>{attrs_eltwise_power});
  auto power_result_var = eltwiseMulOp.getResult();

#if 0
  auto *inst = normalizeOp.getOperation();
  OpBuilder builder(inst);
  auto clonedOp = cast<tpu::NormalizeOp>(builder.clone(*inst));

  auto eltwiseMulOp = rewriter.create<tpu::EltwiseMulOp>(loc, result_type,
      ArrayRef<Value *>{operands_eltwise_power}, ArrayRef<NamedAttribute>{attrs_eltwise_power});
#endif 


#endif

  /* 2. Reduction(using conv2D Op) OP */

  std::vector<Value *> operands_conv;
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

  weightTensorFile_->addTensor<float>(filter_name_conv, weight.data(), filter_type);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(filter_name_conv)));

  auto weight_tensor = rewriter.create<tpu::LoadWeightOp>(loc, filter_type,
      ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});

  operands_conv.push_back(weight_tensor);
  operands_conv.push_back(NoneOp.getResult());
  operands_conv.push_back(NoneOp.getResult());  // quant_scale
  operands_conv.push_back(NoneOp.getResult());  // quant_zeropoint
  operands_conv.push_back(NoneOp.getResult());  // quant_rshift
  operands_conv.push_back(NoneOp.getResult());  // quant_multiplier
  
  // construct OP
  std::vector<NamedAttribute> attrs_conv;
  attrs_conv.push_back(rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(false)));
  attrs_conv.push_back(rewriter.getNamedAttr("padding", rewriter.getStringAttr("VALID")));
  attrs_conv.push_back(rewriter.getNamedAttr("stride_h", rewriter.getI32IntegerAttr(1)));
  attrs_conv.push_back(rewriter.getNamedAttr("stride_w", rewriter.getI32IntegerAttr(1)));
  attrs_conv.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name+"_conv")));

  attrs_conv.push_back(rewriter.getNamedAttr("param",
  tpu::ConvParam::get(
      rewriter.getI32IntegerAttr(stride[0]),
      rewriter.getI32IntegerAttr(stride[1]),
      rewriter.getStringAttr("VALID"),
      rewriter.getI32IntegerAttr(dilation[0]),
      rewriter.getI32IntegerAttr(dilation[1]),
      rewriter.getI32IntegerAttr(g),
      rewriter.getBoolAttr(is_dw),
      rewriter.getBoolAttr(with_bias),
      rewriter.getBoolAttr(false),
      rewriter.getContext())));

  attrs_conv.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  auto convOp = rewriter.create<tpu::Conv2DOp>(
      loc, result_type,
      ArrayRef<Value *>{operands_conv}, ArrayRef<NamedAttribute>{attrs_conv});

  //add convOp after eltwiseMulOp
  //rewriter.replaceOp(eltwiseMulOp, {convOp});
  auto conv_result_var = convOp.getResult();

#if 1

  /* 3. Sqrt OP */
  std::vector<NamedAttribute> attrs_sqrt;
  attrs_sqrt.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name+"_sqrt")));
  attrs_sqrt.push_back(rewriter.getNamedAttr("numerator", rewriter.getF32FloatAttr(1.0)));

  auto sqrt_op = rewriter.create<tpu::SqrtOp>(
      loc, result_type,
      ArrayRef<Value *>{conv_result_var}, ArrayRef<NamedAttribute>{attrs_sqrt});

  //add sqrtOp after convOp
  //rewriter.replaceOp(convOp, {sqrt_op});
  auto sqrt_result_var = sqrt_op.getResult();

  /* 4. Div OP */

  std::vector<NamedAttribute> attrs_div;
  attrs_div.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name+"_Div")));
  attrs_div.push_back(rewriter.getNamedAttr("numerator", rewriter.getF32FloatAttr(1.0)));

  auto div_op = rewriter.create<tpu::DivOp>(
      loc, result_type,
      ArrayRef<Value *>{sqrt_result_var}, ArrayRef<NamedAttribute>{attrs_div});

  //add sqrtOp after convOp
  //rewriter.replaceOp(sqrt_op, {div_op});
  auto div_result_var = div_op.getResult();
 
  /* 5. Eltwise OP(prod) */

  std::vector<Value *> operands_eltwise_mul;

  operands_eltwise_mul.push_back(input_var);
  operands_eltwise_mul.push_back(div_result_var);
  operands_eltwise_mul.push_back(NoneOp.getResult());
  operands_eltwise_mul.push_back(NoneOp.getResult());
  operands_eltwise_mul.push_back(NoneOp.getResult());
  operands_eltwise_mul.push_back(NoneOp.getResult());

  std::vector<NamedAttribute> attrs_eltwise_mul;
  attrs_eltwise_mul.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name+"_eltwise_add")));
  attrs_eltwise_mul.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  auto eltwise_op = rewriter.create<tpu::EltwiseMulOp>(
      loc, result_type,
      ArrayRef<Value *>{operands_eltwise_mul}, ArrayRef<NamedAttribute>{attrs_eltwise_mul});

  //add sqrtOp after convOp
  //rewriter.replaceOp(div_op, {eltwise_op});
  auto eltwise_result_var = eltwise_op.getResult();
 
  /* 6. Scale OP */
  std::vector<Value *> operands_scale;
  operands_scale.push_back(eltwise_result_var);

  auto scale_name = op_name+"_scale_weight";
  auto scale_type = RankedTensorType::get({c}, elementType_);

  weightTensorFile_->addTensor(scale_name, scale, scale_type);
  std::vector<NamedAttribute> scale_weight_attrs;

  scale_weight_attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(scale_name)));

  weight_tensor = rewriter.create<tpu::LoadWeightOp>(loc, scale_type,
      ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{scale_weight_attrs});


  operands_scale.push_back(weight_tensor);
  //no bias , set none
  operands_scale.push_back(NoneOp.getResult());

  // construct scale OP
  std::vector<NamedAttribute> scale_attrs;
  scale_attrs.push_back(rewriter.getNamedAttr(
      "name", rewriter.getStringAttr(op_name+"_scale")));
  auto scale_op = rewriter.create<tpu::ScaleOp>(
      loc, result_type, ArrayRef<Value *>{operands_scale},
      ArrayRef<NamedAttribute>{scale_attrs});
  //add scaleOp after eltwiseOp
  // rewriter.replaceOp(eltwise_op, {scale_op});

  rewriter.replaceOpWithNewOp<tpu::EltwiseMulOp>(
      normalizeOp, result_type,
      ArrayRef<Value *>{operands_eltwise_power}, ArrayRef<NamedAttribute>{attrs_eltwise_power});
#endif
  return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

class DecomposeNormalizePass : public FunctionPass<DecomposeNormalizePass> {
public:
  explicit DecomposeNormalizePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    llvm::StringRef filename;
    Value* weightFileVar;
    fn.walk([&](tpu::LoadFileOp op) {
      weightFileVar = op.getResult();
      filename = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
    });
    auto weightTensorFile = openTensorFile(filename);

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuNormalizePattern>(context, weightTensorFile.get(),weightFileVar);
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

void tpu::NormalizeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  //results.insert<TpuNormalizeOpPattern>(context, nullptr);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDecomposeNormalizePass() {
  return std::make_unique<DecomposeNormalizePass>();
}

static PassRegistration<DecomposeNormalizePass>
    pass("normalize-decompose",
         "Decompose Normalize to ltwise(prod)+conv2D+sqrt+div+eltwise(prod)+scale");
