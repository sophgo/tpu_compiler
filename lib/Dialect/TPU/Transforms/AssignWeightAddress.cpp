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

namespace {

static void transposeConvolutionFilter(std::vector<int8_t> &w, std::vector<int64_t> &s) {
  assert(s.size() == 4);
  int oc = s[0];
  int ic = s[1];
  int ks = s[2] * s[3];
  std::vector<int8_t> w_t(w.size());
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

static void transposeFullyConnectedFilter(std::vector<int8_t> &w, std::vector<int64_t> &s) {
  assert(s.size() == 2);
  int row = s[0];
  int col = s[1];
  std::vector<int8_t> w_t(w.size());
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
    auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);

    auto curPos = weightBinaryFile_->tell();

    if (weightOp.storage() == "INT8") {
      // cast into int8
      std::vector<int8_t> weight_int8(weight->begin(), weight->end());
      // TODO: the transpose was added to workaround a hardware bug
      // TODO: we should remove this transpose if we can verify the tdma tput is the same
      // transpose if this is conv filter weight
      // TODO: this is tricky, we assume any 4 dim weight tensor is a conv filter
      std::vector<int64_t> shape = type.getShape();
      if (shape.size() == 4) {
        transposeConvolutionFilter(weight_int8, shape);
      }
      // TODO: this is tricky, we assume any 2 dim weight tensor is a fc filter
      if (shape.size() == 2) {
        transposeFullyConnectedFilter(weight_int8, shape);
      }
      // TODO: end of workaround transpose
      // pad to alignment
      if ( weight_int8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_int8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_int8.push_back(-128); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_int8.data()),
          weight_int8.size() * sizeof(int8_t));
    } else if (weightOp.storage() == "INT16") {
      // cast into int8
      std::vector<int16_t> weight_int16(weight->begin(), weight->end());
      // TODO: bias are also transposed, should consider removing
      transposeBiasInt16(weight_int16);
      // TODO end of workaround transpose
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
    } else if (weightOp.storage() == "NONE") {
    } else {
      llvm::errs() << tensor_name << " weight storage type "
                   << weightOp.storage() << "\n";
      assert(0 && "not supported weight storage type");
    }

    auto newPos = weightBinaryFile_->tell();
    map_os_ << tensor_name << "," << llvm::format_hex(curPos, 10) << "\n";

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

std::unique_ptr<FunctionPassBase> mlir::createAssignWeightAddressPass() {
  return std::make_unique<AssignWeightAddressPass>();
}

static PassRegistration<AssignWeightAddressPass>
    pass("assign-weight-address",
         "Convert .npz weight file into a .bin file, "
         "and assign weight address to each load weight op");
