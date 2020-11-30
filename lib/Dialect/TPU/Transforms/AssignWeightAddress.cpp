//===- AssignWeightAddress.cpp - assigned weight address ------------------===//
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
// This file is used to assign address to weight
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "assign_weight_address"

using namespace mlir;

namespace {
template<typename OpTy>
struct TpuLoadWeightOpPattern : public RewritePattern {
  TpuLoadWeightOpPattern(MLIRContext *context,
      llvm::raw_fd_ostream *weightBinaryFile, llvm::raw_ostream &map_os,
      size_t alignment, bool compressedWeight)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        weightBinaryFile_(weightBinaryFile),
        map_os_(map_os),
        alignment_(alignment),
        compressedWeight_(compressedWeight) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    auto weightOp = cast<OpTy>(op);
    if (weightOp.offset().hasValue()) {
      // assigned already
      return failure();
    }

    // read the tensor
    auto tensor_name = weightOp.name();
    LLVM_DEBUG(llvm::errs() << "tensor name " << tensor_name << "\n";);

    auto type = weightOp.getResult().getType().template cast<TensorType>();
    assert(weightOp.lowered() && "weight op should be set lowered");
    auto curPos = weightBinaryFile_->tell();
    size_t size = 0;
    if (weightOp.storage() == "INT8") {
      std::vector<int8_t> weight_int8;
      auto weight = wTF->readTensor<int8_t>(tensor_name, type);
      weight_int8.assign(weight->begin(), weight->end());
      size = weight_int8.size();

      // pad to alignment
      if ( weight_int8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_int8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_int8.push_back(-128); // assign a special value for debugging
        }
      }
      auto weightData = reinterpret_cast<const char*>(weight_int8.data());
      weightBinaryFile_->write(weightData, weight_int8.size() *
                               sizeof(int8_t));
    } else if (weightOp.storage() == "UINT8") {
      // UINT8 is used for packed per-channel info or LUT table
      std::vector<uint8_t> weight_uint8;
      auto weight = wTF->readTensor<uint8_t>(tensor_name, type);
      weight_uint8.assign(weight->begin(), weight->end());
      size = weight_uint8.size();

      // pad to alignment
      if ( weight_uint8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_uint8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_uint8.push_back(0xff); // assign a special value for debugging
        }
      }
      auto weightData = reinterpret_cast<const char*>(weight_uint8.data());
      weightBinaryFile_->write(weightData, weight_uint8.size() *
                               sizeof(uint8_t));
    } else if (weightOp.storage() == "INT16") {
      // INT16 is used for bias in INT8 per-tensor mode
      // after lowering, this should be UINT16 already
      llvm_unreachable("unsupported type");
    } else if (weightOp.storage() == "UINT16") {
      // this is NOT BF16 (BF16 uses `BF16` directly)
      // this is for lowered and transposed INT16 bias
      auto weight = wTF->readTensor<uint16_t>(tensor_name, type);
      size = weight->size();
      std::vector<uint16_t> weight_uint16(weight->begin(), weight->end());
      size = weight_uint16.size() * sizeof(uint16_t);

      // pad to alignment
      if ((weight_uint16.size() * sizeof(uint16_t)) % alignment_) {
        size_t remain = (weight_uint16.size() * sizeof(uint16_t)) % alignment_;
        size_t pad = (alignment_ - remain) / sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          // assign a special value for debugging
          weight_uint16.push_back(0xffff);
        }
      }
      weightBinaryFile_->write(
          reinterpret_cast<const char *>(weight_uint16.data()),
          weight_uint16.size() * sizeof(uint16_t));
    } else if (weightOp.storage() == "BF16") {
      std::vector<uint16_t> weight_bf16;
      auto weight = wTF->readTensor<uint16_t>(tensor_name, type);
      weight_bf16.assign(weight->begin(), weight->end());
      size = weight_bf16.size() * sizeof(uint16_t);

      // pad to alignment
      if ( (weight_bf16.size() * sizeof(uint16_t)) % alignment_ ) {
        size_t remain = (weight_bf16.size() * sizeof(uint16_t)) % alignment_;
        size_t pad = (alignment_ - remain) / sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          // assign a special value for debugging
          weight_bf16.push_back(0xffff);
        }
      }
      auto weightData = reinterpret_cast<const char*>(weight_bf16.data());
      weightBinaryFile_->write(weightData, weight_bf16.size() *
                               sizeof(uint16_t));
    } else if (weightOp.storage() == "UINT32") {
      // UINT32 is for lowered Conv Bias
      // 1. Per-Channel (no mulitplier) Conv Bias is supposed to be INT32
      // after transpose, it is stored in striped way (NOT sure yet)
      // 2. BF16 Conv Bias is supposed to be FP32
      // 1880v2 requires storing fp32 into a 2 stripes 16-bit way
      // one stripe for high 16-bit, and one for low 16-bit
      // after the lowering, we store the data as `UINT32`
      std::vector<uint32_t> weight_uint32;
      auto weight = wTF->readTensor<uint32_t>(tensor_name, type);
      weight_uint32.assign(weight->begin(), weight->end());
      size = weight_uint32.size() * sizeof(uint32_t);

      // pad to alignment
      if ( (weight_uint32.size() * sizeof(uint32_t)) % alignment_ ) {
        size_t remain = (weight_uint32.size() * sizeof(uint32_t)) % alignment_;
        size_t pad = (alignment_ - remain) / sizeof(uint32_t);
        for (size_t i = 0; i < pad; ++i) {
          // assign a special value for debugging
          weight_uint32.push_back(0xffffffff);
        }
      }
      auto weightData = reinterpret_cast<const char*>(weight_uint32.data());
      weightBinaryFile_->write(weightData, weight_uint32.size() *
                               sizeof(uint32_t));
    } else if (weightOp.storage() == "FP32") {
      std::vector<float> weight_fp32;
      auto weight = wTF->readTensor<float>(tensor_name, type);
      weight_fp32.assign(weight->begin(), weight->end());
      size = weight_fp32.size() * sizeof(float);

      // pad to alignment
      if ( (weight_fp32.size() * sizeof(float)) % alignment_ ) {
        size_t remain = (weight_fp32.size() * sizeof(float)) % alignment_;
        size_t pad = (alignment_ - remain) / sizeof(float);
        for (size_t i = 0; i < pad; ++i) {
          // assign a special value for debugging
          weight_fp32.push_back(0xffffffff);
        }
      }
      auto weightData = reinterpret_cast<const char*>(weight_fp32.data());
      weightBinaryFile_->write(weightData, weight_fp32.size() * sizeof(float));
    } else if (weightOp.storage() == "NONE") {
      return success();
    } else {
      llvm::errs() << tensor_name << " weight storage type "
                   << weightOp.storage() << "\n";
      assert(0 && "not supported weight storage type");
    }

    auto newPos = weightBinaryFile_->tell();
    map_os_ << tensor_name << "," << llvm::format_hex(curPos, 10) << "\n";

    LLVM_DEBUG(llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 tensor_name.str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";);

    assert(((curPos % alignment_) == 0) && "Expect aligned curPos");
    assert(((newPos % alignment_) == 0) && "Expect aligned newPos");

    // assign the address to weightOp
    weightOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos + (((uint64_t)1) << 40)));

    // Check whether the weight is used by the convolution which indicate it
    // uses the compressed weight.
    if (compressedWeight_) {
      for (auto &use : op->getResult(0).getUses()) {
        auto *useOp = use.getOwner();
        if (auto convOp = dyn_cast<tpu::TL_LW_Conv2DOp>(useOp)) {
          // Weight only, exclude bias.
          if (convOp.filter().getDefiningOp() == op &&
              convOp.compressed_weight().hasValue() &&
              convOp.compressed_weight().getValue()){
            // Mark the weight compressed
            weightOp.setAttr("compressed", rewriter.getBoolAttr(true));
            break;
          }
        }
      }
    }

    return success();
  }

  llvm::raw_fd_ostream *weightBinaryFile_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
  bool compressedWeight_;
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

static llvm::cl::opt<bool> clCompressedWeight(
    "tpu-generate-compressed-weight",
    llvm::cl::desc("Generate the compressed weight"),
    llvm::cl::init(false));

class AssignWeightAddressPass : public mlir::PassWrapper<AssignWeightAddressPass, FunctionPass> {
public:
  explicit AssignWeightAddressPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    // create a bin file
    std::error_code ec;
    assert((clWeightBinFilename != "-") && "no weight bin file specified");
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
    // assign address and generate bin file
    patterns.insert<TpuLoadWeightOpPattern<tpu::LoadWeightOp>,
                    TpuLoadWeightOpPattern<tpu::TL_LG_LoadCoeffOp>
    >(context,
        &weightBinaryFile, weightMapFile->os(), clWeightAlignment,
        clCompressedWeight);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    weightBinaryFile.close();

    if (weightMapFile) {
      weightMapFile->keep();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createAssignWeightAddressPass() {
  return std::make_unique<AssignWeightAddressPass>();
}
