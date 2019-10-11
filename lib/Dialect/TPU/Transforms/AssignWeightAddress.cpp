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
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace {

struct TpuLoadWeightOpPattern : public RewritePattern {
  TpuLoadWeightOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      llvm::raw_fd_ostream *weightBinaryFile, size_t alignment)
      : RewritePattern("tpu.load_weight", 1, context),
        weightTensorFile_(weightTensorFile),
        weightBinaryFile_(weightBinaryFile),
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
    auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);

    // cast into int8
    std::vector<int8_t> weight_int8(weight->begin(), weight->end());

    // pad to alignment
    if (weight->size() % alignment_) {
      size_t pad = alignment_ - (weight->size() % alignment_);
      for (size_t i = 0; i < pad; ++i) {
        weight_int8.push_back(-128); // assign a special value for debugging
      }
    }

    // write to binary file
    auto curPos = weightBinaryFile_->tell();
    weightBinaryFile_->write(reinterpret_cast<const char*>(weight_int8.data()),
        weight_int8.size() * sizeof(int8_t));
    auto newPos = weightBinaryFile_->tell();

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 tensor_name.str().c_str(), weight->size())
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";

    // assign the addres to weightOp
    weightOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  llvm::raw_fd_ostream *weightBinaryFile_;
  size_t alignment_;
};

static llvm::cl::opt<size_t> clWeightAlignment(
    "tpu-weight-address-align",
    llvm::cl::desc("Specify the alignment for weight"),
    llvm::cl::init(16));

class AssignWeightAddressPass : public FunctionPass<AssignWeightAddressPass> {
public:
  explicit AssignWeightAddressPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    //OpBuilder b(fn.getBody());
    llvm::StringRef filename_npz;
    fn.walk<tpu::LoadFileOp>([&](tpu::LoadFileOp op) {
      filename_npz = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename_npz << "\n";
      // NOTE: we didn't assign the LoadFile filename to .bin file
      // keep with npz file so that we can still run interpreter
    });
    auto weightTensorFile = openInputTensorFile(filename_npz);

    // create a bin file
    auto filename_bin = llvm::sys::path::stem(filename_npz.str()).str() + ".bin";
    std::error_code ec;
    llvm::raw_fd_ostream weightBinaryFile(filename_bin, ec);

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuLoadWeightOpPattern>(context, weightTensorFile.get(),
        &weightBinaryFile, clWeightAlignment);
    applyPatternsGreedily(fn, patterns);

    weightBinaryFile.close();
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createAssignWeightAddressPass() {
  return std::make_unique<AssignWeightAddressPass>();
}

static PassRegistration<AssignWeightAddressPass>
    pass("assign-weight-address",
         "Convert .npz weight file into a .bin file, "
         "and assign weight address to each load weight op");
