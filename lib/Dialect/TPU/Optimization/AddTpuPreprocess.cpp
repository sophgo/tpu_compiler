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
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "tpuc/MachineInfo.h"

#define DEBUG_TYPE "add_tpu_preprocess"

using namespace mlir;

static llvm::cl::opt<std::string> clPixelFormat(
    "pixel_format",
    llvm::cl::desc("Set the input pixel format"));

static llvm::cl::opt<bool> clInputAligned(
    "input_aligned",
    llvm::cl::desc("Set if input data is aligned."),
    llvm::cl::init(false));

namespace {

template<typename T>
std::string swapInputChannelOfFilter(
      Operation *op, std::string &filter_name,
      TensorType &filter_type) {

  TensorFile *wTF = getWeightTensorFile(op);
  auto convWeights = wTF->readTensor<T>(filter_name, filter_type);
  // delete the tensor from the weight file
  wTF->deleteTensor<T>(filter_name);

  std::vector<int64_t> shape(filter_type.getShape());
  int64_t size = std::accumulate(std::begin(shape), std::end(shape), 1,
                      std::multiplies<>());
  assert(size == (int64_t)convWeights->size() &&
      "filter size should be equal");

  int64_t oc, ic, frame_size;
  int64_t index = shape.size();
  assert((index == 4 || index == 5) &&
      "filter shape size should be 4 or 5");

  frame_size = shape[index - 1] * shape[index - 2];
  ic = shape[index - 3];
  oc = shape[index - 4];
  if (index == 5) {
    oc *= shape[index - 5];
  }
  std::vector<int> order {2, 1, 0};
  std::vector<T> new_filter(size);

  T *filter = (T *)convWeights->data();
  for (int i = 0; i < oc; ++i) {
    for (int j = 0; j < ic; ++j) {
      assert(order[j] < ic);
      T *in = filter + i * ic * frame_size + order[j] * frame_size;
      T *out =
          (T *)new_filter.data() + i * ic * frame_size + j * frame_size;
      memcpy(out, in, frame_size * sizeof(T));
    }
  }
  std::string tensor_name = filter_name + "_swap_channel";
  wTF->addTensor<T>(tensor_name, &new_filter, filter_type);
  return tensor_name;
}

template<typename OpTy>
struct FoldSwapAxisOpPattern : public RewritePattern {
  FoldSwapAxisOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto swapOp = cast<OpTy>(op);
    auto nextOp = getNextOp(swapOp);
    if (!nextOp) {
      return failure();
    }

    auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(nextOp);
    if (!convOp) {
      return failure();
    }
    if (convOp.param().group().getInt() != 1) {
      return failure();
    }

    // find filter and bias tensor for conv op
    assert(convOp.getNumOperands() == 7 && "Conv2D op should have 7 operands");
    auto filterOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        convOp.getOperand(1).getDefiningOp());
    if (!filterOp) {
      return failure();
    }

    auto filter_name = filterOp.name().str();
    auto filter_type = filterOp.getResult().getType().template cast<TensorType>();

    std::string new_name;
    if (filter_type.getElementType().isBF16()) {
      new_name = swapInputChannelOfFilter<uint16_t>(convOp, filter_name, filter_type);
    } else {
      new_name = swapInputChannelOfFilter<float>(convOp, filter_name, filter_type);
    }

    auto wfV = getWeightFileValue(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
        rewriter.getStringAttr(new_name)));
    attrs.push_back(rewriter.getNamedAttr("storage",
        rewriter.getStringAttr(filterOp.storage().str())));
    auto newFilterOp = rewriter.create<tpu::LoadWeightOp>(
        filterOp.getLoc(), filter_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{attrs});

    rewriter.replaceOp(filterOp.getResult().getDefiningOp(),
                       {newFilterOp.getResult()});
    rewriter.replaceOp(op, {swapOp.getOperand()});
    return success();
  }
};

class AddTpuPreprocessPass
    : public mlir::PassWrapper<AddTpuPreprocessPass,
                               FunctionPass> {
public:
  void runOnFunction() override {
    MInfo::getChipInfo(getFunction());
    assert(MInfo::version && "refer to set-chip");

    auto *context = &getContext();
    auto builder = OpBuilder(context);
    auto fn = getFunction();

    std::vector<mlir::Type> returnTypes;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returnTypes.push_back(returnOp->getOperand(i).getType());
    }

    std::vector<mlir::Type> argumentTypes;

    std::map<std::string,
             std::pair<std::string, std::string>> attributes_map = {
      {"RGB_PLANAR",    {"rgb", "nchw"}},
      {"RGB_PACKED",    {"rgb", "nhwc"}},
      {"BGR_PLANAR",    {"bgr", "nchw"}},
      {"BGR_PACKED",    {"bgr", "nhwc"}},
      {"GRAYSCALE",     {"bgr", "nchw"}},
      {"YUV420_PLANAR", {"bgr", "nchw"}}
    };

    fn.walk([&](tpu::InputOp inputOp) {
      // get attributes of input
      StringRef quantType = "INT8";
      auto nextOp = getNextOp(inputOp);
      if (nextOp) {
        if (auto quantOp = dyn_cast<tpu::QuantOp>(nextOp)) {
          quantType = quantOp.to();
          quantOp.replaceAllUsesWith(inputOp.getResult());
          quantOp.erase();
        }
      }
      auto threshold = getOpThreshold(inputOp);
      auto name = mlir::getOpName(inputOp).str();
      auto preprocess = inputOp.preprocessAttr();
      auto pixel_format = clPixelFormat.length() ? clPixelFormat:
                          preprocess.pixel_format().getValue().str();
      auto resize_dims = preprocess.resize_dims();
      auto aligned = clInputAligned.getValue();
      auto channel_order = preprocess.channel_order().getValue();
      auto model_shape = getTensorShape(inputOp.getResult());
      getNCHW(model_shape, n, c, h, w);
      std::vector<int64_t> dims;
      for (auto dim : resize_dims.getAsValueRange<IntegerAttr>()) {
        dims.push_back(dim.getSExtValue());
      }
      resize_h = dims[0];
      resize_w = dims[1];

      auto color = std::get<0>(attributes_map[pixel_format]);
      auto layout = std::get<1>(attributes_map[pixel_format]);
      bool swap_channel = (color != channel_order) ? true : false;
      llvm::errs() << "pixel_format:" << pixel_format
                   << ", color:" << color
                   << ", layout:" << layout
                   << ", swap_channel:" << swap_channel
                   << ", aligned:" << aligned
                   << "\n";

      std::vector<Operation *> uses;
      for (auto &use : inputOp.getResult().getUses()) {
        auto opd = use.getOwner();
        uses.push_back(opd);
      }

      // set the real shape of function's args.
      std::vector<int64_t> arg_shape {n, c, resize_h, resize_w};
      std::vector<int64_t> input_shape {n, c, resize_h, resize_w};
      if (layout == "nhwc") {
        arg_shape[1] = resize_h;
        arg_shape[2] = resize_w;
        arg_shape[3] = c;
        if (aligned) {
          input_shape[1] = 1;
          input_shape[2] = resize_h;
          input_shape[3] = align_up(resize_w * c, 32);
        } else {
          input_shape[1] = resize_h;
          input_shape[2] = resize_w;
          input_shape[3] = c;
        }
      } else { // "nchw"
        if (pixel_format == "YUV420_PLANAR") {
          input_shape[1] = 1;
          input_shape[2] = 1;
          input_shape[3] = yuv420_size(1, c, resize_h, resize_w);
          aligned = true; // currently only support aligned yuv420 input.
        } else if (aligned) {
          input_shape[1] = c;
          input_shape[2] = resize_h;
          input_shape[3] = align_up(resize_w, 32);
        }
      }
      auto arg_type = this->getTensorType(builder, arg_shape, "UINT8");
      inputOp.getOperand().setType(arg_type);
      setOpThreshold(inputOp, 255);
      setOpQuantParamType(inputOp, "THRESHOLD");
      setOpQuant(inputOp, "UINT8");
      argumentTypes.push_back(arg_type);

      // change the shape of inputOp
      auto input_type = this->getTensorType(builder, input_shape, "UINT8");
      inputOp->setAttr("name", builder.getStringAttr(name + "_raw"));
      inputOp->setAttr("preprocess",
          tpu::PreprocessParam::get(
              builder.getStringAttr(pixel_format),
              builder.getBoolAttr(aligned),
              preprocess.resize_dims(),
              preprocess.keep_aspect_ratio(),
              preprocess.channel_order(),
              builder.getF32ArrayAttr({1.0f, 1.0f, 1.0f}),
              builder.getF32ArrayAttr({0.0f, 0.0f, 0.0f}),
              builder.getContext()));
      inputOp.getResult().setType(input_type);

      mlir::Value currentOp = inputOp.getResult();
      builder.setInsertionPointAfter(inputOp);

      // do unalign
      if (aligned) {
        currentOp = this->insertCscOp(builder, name, currentOp,
                                      pixel_format, aligned, arg_shape);
      }
      // create transpose Op if need
      if (layout == "nhwc") {
        currentOp = this->insertTransposeOp(builder, name, currentOp);
      }
      // create cropOp
      if (resize_h != h || resize_w != w) {
        currentOp = this->insertCropOp(builder, name, currentOp);
      }
      // create scale op
      if (quantType == "INT8") {
        // scale by lut
        currentOp = this->insertScaleLutOp(
            builder, name, currentOp, preprocess, threshold, swap_channel);
      } else {
        currentOp = this->insertDequantOp(builder, name, currentOp);
        currentOp = this->insertScaleOp(builder, name, currentOp, preprocess,
                                        threshold, swap_channel);
      }

      if (swap_channel) {
        currentOp = this->insertSwapAxisOp(builder, name, currentOp, threshold,
                                           quantType.str());
      }
      // update operand of all inputOp's uses
      for (auto use_op : uses) {
        for (int i = 0; i < (int)use_op->getNumOperands(); i++) {
          if (use_op->getOperand(i) == inputOp.getResult()) {
            use_op->setOperand(i, currentOp);
            llvm::errs() << "set operand\n";
          }
        }
      }
    });

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    auto fnType = builder.getFunctionType(
          llvm::ArrayRef<mlir::Type>{argumentTypes},
          llvm::ArrayRef<mlir::Type>{returnTypes});
    fn.setType(fnType);

    OwningRewritePatternList patterns;
    patterns.insert<
        FoldSwapAxisOpPattern<tpu::SwapChannelOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  int64_t n, c, h, w;
  int64_t resize_h;
  int64_t resize_w;

private:
  inline int align_up(int x, int n) {
    return ((x + n - 1) / n) * n;
  }

  int yuv420_size(int n, int c, int h, int w) {
    assert(c == 3);
    int y_w_aligned = align_up(w, 32);
    int uv_w_aligned = align_up(w / 2, 32);
    int u = align_up(h * y_w_aligned, 0x1000);
    int v = align_up(u + h / 2 * uv_w_aligned, 0x1000);
    int n_stride = align_up(v + h / 2 * uv_w_aligned, 0x1000);
    return n * n_stride;
  }

  RankedTensorType getTensorType(OpBuilder &builder, const std::vector<int64_t> &shape,
                                 const std::string &quantType) {
    Type eltType;
    if (quantType == "INT8") {
      eltType = IntegerType::get(builder.getContext(), 8, IntegerType::Signed);
    } else if (quantType == "UINT8") {
      eltType = IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned);
    } else if (quantType == "BF16") {
      eltType = FloatType::getBF16(builder.getContext());
    } else {
      eltType = FloatType::getF32(builder.getContext());
    }
    return RankedTensorType::get(shape, eltType);
  }

  Value insertCscOp(OpBuilder &builder, std::string &name, Value opd,
                    std::string &pixel_format, bool aligned,
                    std::vector<int64_t> &shape) {
    std::string name_ = name + "_preprocess_csc";
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("pixel_format",
        builder.getStringAttr(pixel_format)));
    attrs.push_back(builder.getNamedAttr("aligned",
        builder.getBoolAttr(aligned)));
    attrs.push_back(builder.getNamedAttr("quant",
        getDefaultQuantParam(builder)));

    auto type = this->getTensorType(builder, shape, "UINT8");
    auto newOp = builder.create<tpu::CscOp>(
        opd.getLoc(), type, ArrayRef<Value>{opd},
        ArrayRef<NamedAttribute>{attrs});

    setOpThreshold(newOp, 255);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, "UINT8");
    return newOp;
  }

  Value insertTransposeOp(OpBuilder &builder, std::string &name, Value opd) {
    auto name_ = name + "_preprocess_tranpose";
    std::vector<int> orders{0, 3, 1, 2};

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("quant",
        getDefaultQuantParam(builder)));
    attrs.push_back(builder.getNamedAttr("order0",
        builder.getI32IntegerAttr(orders[0])));
    attrs.push_back(builder.getNamedAttr("order1",
        builder.getI32IntegerAttr(orders[1])));
    attrs.push_back(builder.getNamedAttr("order2",
        builder.getI32IntegerAttr(orders[2])));
    attrs.push_back(builder.getNamedAttr("order3",
        builder.getI32IntegerAttr(orders[3])));

    auto type = this->getTensorType(builder, {n, c, resize_h, resize_w}, "UINT8");
    auto newOp = builder.create<tpu::PermuteOp>(
        opd.getLoc(), type, ArrayRef<Value>{opd},
        ArrayRef<NamedAttribute>{attrs});

    setOpThreshold(newOp, 255);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, "UINT8");
    return newOp;
  }

  Value insertCropOp(OpBuilder &builder, std::string &name, Value opd) {
    std::string name_ = name + "_preprocess_crop";
    int start_h = resize_h / 2 - h / 2;
    int start_w = resize_w / 2 - w / 2;
    std::vector<int> crop_offset {0, 0, start_h, start_w};
    std::vector<int> crop_shape {(int)n, (int)c, (int)h, (int)w};

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("quant",
        getDefaultQuantParam(builder)));
    attrs.push_back(builder.getNamedAttr("crop_shape",
        builder.getI32ArrayAttr(crop_shape)));
    attrs.push_back(builder.getNamedAttr("crop_offset",
        builder.getI32ArrayAttr(crop_offset)));

    auto type = this->getTensorType(builder, {n, c, h, w}, "UINT8");
    auto newOp = builder.create<tpu::CropOp>(
        opd.getLoc(), type, ArrayRef<Value>{opd},
        ArrayRef<NamedAttribute>{attrs});

    setOpThreshold(newOp, 255);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, "UINT8");
    return newOp;
  }

  Value insertScaleOp(OpBuilder &builder, std::string &name, Value opd,
                      const tpu::PreprocessParam &param, float threshold,
                      bool swap_channel) {
    std::string name_ = name + "_preprocess_scale";

    auto scale_type = this->getTensorType(builder, {c, 1, 1, 1, 1}, "NONE");
    auto bias_type = this->getTensorType(builder, {c}, "NONE");

    std::vector<float> scale;
    std::vector<float> bias;
    for (auto s : param.scale().getAsValueRange<FloatAttr>()) {
      scale.push_back(s.convertToFloat());
    }
    for (auto m : param.mean().getAsValueRange<FloatAttr>()) {
      bias.push_back(-1 * m.convertToFloat());
    }
    llvm::errs() << "scale:";
    for (auto s : scale)
      llvm::errs() << " " << s;
    llvm::errs() << "\n";
    llvm::errs() << "bias:";
    for (auto b : bias)
      llvm::errs() << " " << b;
    llvm::errs() << "\n";

    // swap op do after scale(because it can be fuesd with first conv)
    // change weight order here
    if (c == 3 && swap_channel) {
      std::swap(scale[0], scale[2]);
      std::swap(bias[0], bias[2]);
    }

    TensorFile *wTF = getWeightTensorFile(opd.getDefiningOp());
    Value wfV = getWeightFileValue(opd.getDefiningOp());

    wTF->addTensor<float>(name_ + "_0", scale.data(), scale_type);
    wTF->addTensor<float>(name_ + "_1", bias.data(), bias_type);

    std::vector<NamedAttribute> scale_attrs, bias_attrs;
    scale_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_ + "_0")));
    bias_attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_ + "_1")));
    auto scale_op = builder.create<tpu::LoadWeightOp>(
        opd.getLoc(), scale_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{scale_attrs});
    auto bias_op = builder.create<tpu::LoadWeightOp>(
        opd.getLoc(), bias_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{bias_attrs});

    std::vector<Value> operands;
    operands.push_back(opd);
    operands.push_back(scale_op);
    operands.push_back(bias_op);

    auto NoneOp = builder.create<tpu::NoneOp>(
        opd.getLoc(), builder.getNoneType());
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult());  // quant_multiplier

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("param",
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
            builder.getI32IntegerAttr((int)c),
            builder.getBoolAttr(true),
            builder.getBoolAttr(true),
            builder.getBoolAttr(false),
            builder.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
            builder.getI32IntegerAttr(0), //pad_value
            builder.getContext())));
    attrs.push_back(
        builder.getNamedAttr("quant", getDefaultQuantParam(builder)));

    auto type = this->getTensorType(builder, {n, c, h, w}, "BF16");
    auto newOp = builder.create<tpu::Conv2DOp>(
        opd.getLoc(), type, ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    setOpThreshold(newOp, threshold);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, "BF16");
    newOp.quantizeBf16();
    return newOp;
  }

  Value insertScaleLutOp(OpBuilder &builder, std::string &name, Value opd,
                      const tpu::PreprocessParam &param, float threshold,
                      bool swap_channel) {
    std::string name_ = name + "_preprocess_scale_lut";
    float qscale = 128.0f / threshold;

    std::vector<float> scale;
    std::vector<float> bias;
    for (auto s : param.scale().getAsValueRange<FloatAttr>()) {
      scale.push_back(s.convertToFloat());
    }
    for (auto m : param.mean().getAsValueRange<FloatAttr>()) {
      bias.push_back(-1 * m.convertToFloat());
    }
    if (scale.empty()) {
      scale.assign(3, 1.0);
    }
    if (bias.empty()) {
      bias.assign(3, 0.0);
    }
    if (swap_channel) {
      // keep order bgr
      std::swap(scale[0], scale[2]);
      std::swap(bias[0], bias[2]);
    }
    for (int i = 0; i < 3; i++) {
      scale[i] *= qscale;
      bias[i] *= qscale;
    }

    std::vector<Value> operands;
    operands.push_back(opd);
    auto NoneOp = builder.create<tpu::NoneOp>(opd.getLoc(), builder.getNoneType());
    operands.push_back(NoneOp.getResult()); // table

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("scale",builder.getF32ArrayAttr(ArrayRef<float>(scale))));
    attrs.push_back(builder.getNamedAttr("bias",builder.getF32ArrayAttr(ArrayRef<float>(bias))));
    attrs.push_back(builder.getNamedAttr("quant", getDefaultQuantParam(builder)));

    auto type = this->getTensorType(builder, {n, c, h, w}, "INT8");
    auto newOp = builder.create<tpu::ScaleLutOp>(
        opd.getLoc(), type, ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});
    setOpThreshold(newOp, threshold);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, "INT8");
    newOp.quantizeInt8();
    return newOp;
  }

  // swapaxis, rgb to bgr or bgr to rgb
  Value insertSwapAxisOp(OpBuilder &builder, std::string &name, Value opd,
                        float threshold, std::string quant_type) {
    std::string name_ = name + "_preprocess_swapaxis";

    std::vector<NamedAttribute> attrs;
    std::vector<int> orders {2, 1, 0};

    attrs.push_back(
        builder.getNamedAttr("name", builder.getStringAttr(name_)));
    attrs.push_back(
        builder.getNamedAttr("quant", getDefaultQuantParam(builder)));
    attrs.push_back(
        builder.getNamedAttr("channel_order",
            builder.getI32ArrayAttr(ArrayRef<int32_t>({orders}))));
    // we only accept first input to IR, second input shape will be attribute.
    auto type = this->getTensorType(builder, {n, c, h, w}, quant_type);
    auto newOp = builder.create<tpu::SwapChannelOp>(
        opd.getLoc(), type, ArrayRef<Value>{opd},
        ArrayRef<NamedAttribute>{attrs});

    setOpThreshold(newOp, threshold);
    setOpQuantParamType(newOp, "THRESHOLD");
    setOpQuant(newOp, quant_type);
    return newOp;
  }

  Value insertDequantOp(OpBuilder &builder, std::string &name, Value opd) {
    std::string name_ = name + "_preprocess_dequant_bf16";
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name_)));
    attrs.push_back(builder.getNamedAttr("from",
        builder.getStringAttr("UINT8")));
    attrs.push_back(builder.getNamedAttr("to",
        builder.getStringAttr("BF16")));
    attrs.push_back(builder.getNamedAttr("scale",
        builder.getF32FloatAttr(1.0)));
    attrs.push_back(builder.getNamedAttr("zero_point",
        builder.getI32IntegerAttr(0)));

    auto type = this->getTensorType(builder, {n, c, h, w}, "BF16");
    auto quantOp = builder.create<tpu::QuantOp>(opd.getLoc(), type,
        ArrayRef<Value>{opd}, ArrayRef<NamedAttribute>{attrs});
    return quantOp;
  }

};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createAddTpuPreprocessPass() {
  return std::make_unique<AddTpuPreprocessPass>();
}
