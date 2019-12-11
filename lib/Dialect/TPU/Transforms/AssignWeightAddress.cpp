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
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

template<typename T>
static void transposeConvolutionFilter(std::vector<T> &w, std::vector<int64_t> &s) {
  assert(s.size() == 4);
  int oc = s[0];
  int ic = s[1];
  int ks = s[2] * s[3];
  std::vector<T> w_t(w.size());
  if (ks == 1) {
    return;
  } else {
    // for other conv, transpose ic <-> kh*kw
    for (int i = 0; i < oc; i++) {
      for (int j = 0; j < ic; j++) {
        for (int k = 0; k < ks; k++) {
          w_t[i * ic * ks + k * ic + j] = w[i * ic * ks + j * ks + k];
        }
      }
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

template<typename T>
static void transposeFullyConnectedFilter(std::vector<T> &w, std::vector<int64_t> &s) {
  assert(s.size() == 2);
  int row = s[0];
  int col = s[1];
  std::vector<T> w_t(w.size());
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = w[i * col  + j];
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

static void transposeBiasInt16(std::vector<int16_t> &w_int16) {
  int8_t *ptr = reinterpret_cast<int8_t *>(w_int16.data());
  std::vector<int8_t> w(ptr, ptr + w_int16.size() * sizeof(int16_t));
  std::vector<int8_t> w_t(w.size());
  for (size_t i = 0; i < w_int16.size(); i++) {
    for (size_t j = 0; j < 2; j++) {
      w_t[j * w_int16.size() + i] = w[i * 2 + j];
    }
  }
  memcpy(ptr, w_t.data(), w_t.size());
}

namespace {

struct TpuConv2DOpPattern : public RewritePattern {
  TpuConv2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.conv_2d", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    if (convOp.quant() != "INT8_MULTIPLIER") {
      return matchFailure();
    }

    // for per-channel multiplier mode
    // we need to pack rshift, multiplier and bias into one weight
    // layout is: [bias_4byte] multiplier_4byte rshift_1byte
    // with bias 9 bytes, w/o bias 5 bytes, for each output channel
    std::string op_name = convOp.name().getValue().str();
    llvm::errs() << op_name
                 << ", pack Conv2D rshift multipler and bias\n";
    bool has_bias = false;
    if (convOp.getNumOperands() == 5) {
      has_bias = true;
      llvm::errs() << "  with bias\n";
    } else if (convOp.getNumOperands() == 4) {
      llvm::errs() << "  no bias\n";
    } else {
      llvm::errs() << "  already processed\n";
      assert(convOp.getNumOperands() == 3);
      return matchFailure();
    }

    std::vector<std::unique_ptr<std::vector<float> > > weights;
    for (unsigned i = 2; i < convOp.getNumOperands(); ++i) {
      auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(i)->getDefiningOp());
      assert(weightOp);
      assert(weightOp.name().hasValue());
      auto tensor_name = weightOp.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weightOp.getResult()->getType().cast<TensorType>();
      auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);
      weights.push_back(std::move(weight));
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
      // mark weightOp as no storage
      weightOp.setAttr("storage", rewriter.getStringAttr("NONE"));
    }

    float *bias = nullptr;
    float *multiplier = nullptr;
    float *rshift = nullptr;
    if (has_bias) {
      assert(weights.size() == 3);
      bias = (float *)weights[0]->data();
      multiplier = (float *)weights[2]->data();
      rshift = (float *)weights[1]->data();
    } else {
      assert(weights.size() == 2);
      multiplier = (float *)weights[1]->data();
      rshift = (float *)weights[0]->data();
    }

    // pack the weights
    auto filter_type = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t oc;
    auto g = convOp.group().getLimitedValue();
    if (g != 1) {
      assert(filter_shape.size() == 5);
      oc = filter_shape[0] * filter_shape[1];
    } else {
      assert(filter_shape.size() == 4);
      oc = filter_shape[0];
    }

    int64_t isz = has_bias ? 9 : 5;
    std::vector<float> newWeight(oc * isz);
    std::vector<int64_t> newWeightShape = std::vector<int64_t>{oc, 1, isz};

    float *ptr = (float *)newWeight.data();;
    for (int i = 0; i < oc; i++) {
      if (has_bias) {
        uint32_t val = (uint32_t)bias[i];
        *ptr = (float)(val & 0xff);
        ptr++;
        *ptr = (float)((val >> 8) & 0xff);
        ptr++;
        *ptr = (float)((val >> 16) & 0xff);
        ptr++;
        *ptr = (float)((val >> 24) & 0xff);
        ptr++;
      }

      {
        uint32_t val = (uint32_t)multiplier[i];
        *ptr = (float)(val & 0xff);
        ptr++;
        *ptr = (float)((val >> 8) & 0xff);
        ptr++;
        *ptr = (float)((val >> 16) & 0xff);
        ptr++;
        *ptr = (float)((val >> 24) & 0xff);
        ptr++;
      }

      {
        uint8_t val = (uint8_t)rshift[i];
        *ptr = (float)val;
        ptr++;
      }
    }

    // update the convOp
    std::vector<Value *> newOperands;
    newOperands.push_back(convOp.getOperand(0));
    newOperands.push_back(convOp.getOperand(1));
    // add new weight
    {
      // find weightFileVar
      auto oneWeightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(1)->getDefiningOp());
      auto weightFileVar = oneWeightOp.getOperand();

      auto tensor_name = op_name + "_per_channel";
      auto type = RankedTensorType::get(newWeightShape,
          FloatType::getF32(rewriter.getContext()));
      llvm::errs() << "  newWeight : " << tensor_name << "\n";
      weightTensorFile_->addTensor<float>(tensor_name, &newWeight, type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
      auto newWeightOp = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(newWeightOp);
    }

    // replace the convOp
    auto origAttrs = convOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        convOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

struct TpuLoadWeightOpPattern : public RewritePattern {
  TpuLoadWeightOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      llvm::raw_fd_ostream *weightBinaryFile, llvm::raw_ostream &map_os,
      size_t alignment)
      : RewritePattern("tpu.load_weight", 1, context),
        weightTensorFile_(weightTensorFile),
        weightBinaryFile_(weightBinaryFile),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto weightOp = cast<tpu::LoadWeightOp>(op);
    if (weightOp.offset().hasValue()) {
      // assigned already
      return matchFailure();
    }

    // read the tensor
    auto tensor_name = weightOp.name().getValue();
    auto type = weightOp.getResult()->getType().cast<TensorType>();
    auto curPos = weightBinaryFile_->tell();
    size_t size;
    if (weightOp.storage() == "INT8") {
      auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);
      size = weight->size();
      // cast into int8
      std::vector<int8_t> weight_int8(weight->begin(), weight->end());

      // hw design needs transpose on filters
      // transpose if this is conv filter weight
      // TODO: this is tricky, we assume any 4 dim weight tensor is a conv filter
      std::vector<int64_t> shape = type.getShape();
      if (shape.size() == 4) {
        transposeConvolutionFilter<int8_t>(weight_int8, shape);
      }
      // TODO: this is tricky, we assume any 2 dim weight tensor is a fc filter
      if (shape.size() == 2) {
        transposeFullyConnectedFilter<int8_t>(weight_int8, shape);
      }

      // pad to alignment
      if ( weight_int8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_int8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_int8.push_back(-128); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_int8.data()),
          weight_int8.size() * sizeof(int8_t));
    } else if (weightOp.storage() == "UINT8") {
      auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);
      size = weight->size();
      // cast into int8
      std::vector<uint8_t> weight_uint8(weight->begin(), weight->end());
      // pad to alignment
      if ( weight_uint8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_uint8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_uint8.push_back(0xff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_uint8.data()),
          weight_uint8.size() * sizeof(uint8_t));
    } else if (weightOp.storage() == "INT16") {
      auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);
      size = weight->size();
      // cast into int8
      std::vector<int16_t> weight_int16(weight->begin(), weight->end());
      // bias are also transposed
      transposeBiasInt16(weight_int16);
      // pad to alignment
      if ( weight_int16.capacity() % alignment_ ) {
        size_t pad = ( alignment_ - ( weight_int16.capacity() % alignment_ ) )
                     / sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          weight_int16.push_back(-32768); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_int16.data()),
          weight_int16.size() * sizeof(int16_t));
    } else if (weightOp.storage() == "BF16") {
      auto weight = weightTensorFile_->readTensor<uint16_t>(tensor_name, type);
      size = weight->size();
      std::vector<uint16_t> weight_bf16(weight->begin(), weight->end());

      // hw design needs transpose on filters
      // transpose if this is conv filter weight
      // TODO: this is tricky, we assume any 4 dim weight tensor is a conv filter
      std::vector<int64_t> shape = type.getShape();
      if (shape.size() == 4) {
        transposeConvolutionFilter<uint16_t>(weight_bf16, shape);
      }
      // TODO: this is tricky, we assume any 2 dim weight tensor is a fc filter
      if (shape.size() == 2) {
        transposeFullyConnectedFilter<uint16_t>(weight_bf16, shape);
      }
      // TODO: this is even more tricky (FIXME asap), assume 1 dim tensor is bias
      // for bm1880v2, bias is fp32, but store as 2 separate stripe
      // one for high 16-bit, one for low-16 bit
      // we use the quantized bf16 bias as high 16-bit, and add a zero stripe low 16-bit
      if (shape.size() == 1) {
        size_t sz = weight_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          weight_bf16.push_back(0x0000);
        }
      }

      // pad to alignment
      if ( weight_bf16.capacity() % alignment_ ) {
        size_t pad = ( alignment_ - ( weight_bf16.capacity() % alignment_ ) )
                     / sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          weight_bf16.push_back(0xffff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_bf16.data()),
          weight_bf16.size() * sizeof(uint16_t));
    } else if (weightOp.storage() == "NONE") {
      return matchSuccess();
    } else {
      llvm::errs() << tensor_name << " weight storage type "
                   << weightOp.storage() << "\n";
      assert(0 && "not supported weight storage type");
    }

    auto newPos = weightBinaryFile_->tell();
    map_os_ << tensor_name << "," << llvm::format_hex(curPos, 10) << "\n";

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 tensor_name.str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";

    // assign the address to weightOp
    weightOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  llvm::raw_fd_ostream *weightBinaryFile_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};

static llvm::cl::opt<size_t> clWeightAlignment(
    "tpu-weight-address-align",
    llvm::cl::desc("Specify the alignment for weight"),
    llvm::cl::init(16));

static llvm::cl::opt<std::string> clWeightMapFilename(
    "tpu-weight-map-filename",
    llvm::cl::desc("record weight offset with its name into a csv map file"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> clWeightBinFilename(
    "tpu-weight-bin-filename",
    llvm::cl::desc("weight bin filename"),
    llvm::cl::init("-"));

class AssignWeightAddressPass : public FunctionPass<AssignWeightAddressPass> {
public:
  explicit AssignWeightAddressPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    //OpBuilder b(fn.getBody());

    // find tensor filename
    llvm::StringRef filename_npz;
    fn.walk([&](tpu::LoadFileOp op) {
      filename_npz = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename_npz << "\n";
      // NOTE: we didn't assign the LoadFile filename to .bin file
      // keep with npz file so that we can still run interpreter
    });
    auto weightTensorFile = openTensorFile(filename_npz);

    // create a bin file
    std::error_code ec;
    assert(clWeightBinFilename != "-");
    llvm::raw_fd_ostream weightBinaryFile(clWeightBinFilename, ec);

    // create a map file
    std::unique_ptr<llvm::ToolOutputFile> weightMapFile = nullptr;
    if (clWeightMapFilename != "-") {
      std::string errorMessage;
      weightMapFile = openOutputFile(clWeightMapFilename, &errorMessage);
      if (!weightMapFile) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
    }

    OwningRewritePatternList patterns;
    auto *context = &getContext();

    // merge conv rshift/multiplier/bias into one weight first
    patterns.insert<TpuConv2DOpPattern>(context, weightTensorFile.get());
    applyPatternsGreedily(fn, patterns);
    patterns.clear();

    // don't keep this file by default, remove comment for debug only
    // weightTensorFile->keep();

    // TODO: apply transpose in pattern match

    // assign address and generate bin file
    patterns.insert<TpuLoadWeightOpPattern>(context, weightTensorFile.get(),
        &weightBinaryFile, weightMapFile->os(), clWeightAlignment);
    applyPatternsGreedily(fn, patterns);

    weightBinaryFile.close();

    if (weightMapFile) {
      weightMapFile->keep();
    }
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignWeightAddressPass() {
  return std::make_unique<AssignWeightAddressPass>();
}

static PassRegistration<AssignWeightAddressPass>
    pass("assign-weight-address",
         "Convert .npz weight file into a .bin file, "
         "and assign weight address to each load weight op");
