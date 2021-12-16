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
#include "mlir/IR/BuiltinTypes.h"
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
#include <openssl/md5.h>
#include <list>
#include <map>
#include <unordered_map>
#include <memory>
#include <set>
#include <iostream>
#include <fstream>
#include <string>

#define DEBUG_TYPE "assign_weight_address"

using namespace mlir;

namespace {

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

static llvm::cl::opt<bool> clAppendWeight(
    "tpu-append-weight",
    llvm::cl::desc("append weight to weight bin file"),
    llvm::cl::init(false));


template<typename T>
static bool isRedundantWeight(Operation *op, std::vector<T> &weight_vec,
                        std::map<std::string, uint64_t> &map,
                        uint64_t cur_pos, std::string &md5) {
  MD5_CTX ctx;
  MD5_Init(&ctx);
  auto size = weight_vec.size() * sizeof(T);
  MD5_Update(&ctx, weight_vec.data(), size);
  unsigned char res[16];
  MD5_Final(res, &ctx);

  std::string s;
  llvm::raw_string_ostream os(s);
  for(int i=0; i < 16; ++i)
    os << llvm::format_hex_no_prefix((int)res[i], 2);
  md5 = os.str();

  if (map.find(md5) != map.end()) {
    return true;
  }
  map[md5] = cur_pos;
  return false;
}

template<typename OpTy>
struct TpuLoadWeightOpPattern : public RewritePattern {
  TpuLoadWeightOpPattern(MLIRContext *context,
      std::unique_ptr<std::fstream> &weightBinFile,
      std::unique_ptr<std::fstream> &weightMapFile,
      std::map<std::string, uint64_t> &map,
      size_t alignment)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        bin_os_(weightBinFile),
        map_os_(weightMapFile),
        md5AddrMap_(map),
        alignment_(alignment) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    auto weightOp = cast<OpTy>(op);
    if (weightOp.offset().hasValue()) {
      // assigned already
      return failure();
    }

    // read the tensor
    auto tensor_name = weightOp.name().str();
    LLVM_DEBUG(llvm::errs() << "tensor name " << tensor_name << "\n";);

    auto type = weightOp.getResult().getType().template cast<TensorType>();
    assert(weightOp.lowered() && "weight op should be set lowered");
    auto curPos = (uint64_t)bin_os_->tellp();
    size_t size = 0;
    std::string md5;
    bool isRedundant = false;

    if (weightOp.storage() == "INT8") {
      std::vector<int8_t> weight_int8;
      auto weight = wTF->readTensor<int8_t>(tensor_name, type);
      weight_int8.assign(weight->begin(), weight->end());
      size = weight_int8.size();

      isRedundant = isRedundantWeight<int8_t>(op, weight_int8, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
        // pad to alignment
        if ( weight_int8.size() % alignment_ ) {
          size_t pad = alignment_ - (weight_int8.size() % alignment_);
          for (size_t i = 0; i < pad; ++i) {
            weight_int8.push_back(-128); // assign a special value for debugging
          }
        }
        auto weightData = reinterpret_cast<const char*>(weight_int8.data());
        bin_os_->write(weightData, weight_int8.size() *
                                sizeof(int8_t));
      }
    } else if (weightOp.storage() == "UINT8") {
      // UINT8 is used for packed per-channel info or LUT table
      std::vector<uint8_t> weight_uint8;
      auto weight = wTF->readTensor<uint8_t>(tensor_name, type);
      weight_uint8.assign(weight->begin(), weight->end());
      size = weight_uint8.size();

      isRedundant = isRedundantWeight<uint8_t>(op, weight_uint8, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
        // pad to alignment
        if ( weight_uint8.size() % alignment_ ) {
          size_t pad = alignment_ - (weight_uint8.size() % alignment_);
          for (size_t i = 0; i < pad; ++i) {
            weight_uint8.push_back(0xff); // assign a special value for debugging
          }
        }
        auto weightData = reinterpret_cast<const char*>(weight_uint8.data());
        bin_os_->write(weightData, weight_uint8.size() * sizeof(uint8_t));
      }
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

      isRedundant = isRedundantWeight<uint16_t>(op, weight_uint16, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
        // pad to alignment
        if ((weight_uint16.size() * sizeof(uint16_t)) % alignment_) {
          size_t remain = (weight_uint16.size() * sizeof(uint16_t)) % alignment_;
          size_t pad = (alignment_ - remain) / sizeof(uint16_t);
          for (size_t i = 0; i < pad; ++i) {
            // assign a special value for debugging
            weight_uint16.push_back(0xffff);
          }
        }
        bin_os_->write(
            reinterpret_cast<const char *>(weight_uint16.data()),
            weight_uint16.size() * sizeof(uint16_t));
      }
    } else if (weightOp.storage() == "BF16") {
      std::vector<uint16_t> weight_bf16;
      auto weight = wTF->readTensor<uint16_t>(tensor_name, type);
      weight_bf16.assign(weight->begin(), weight->end());
      size = weight_bf16.size() * sizeof(uint16_t);

      isRedundant = isRedundantWeight<uint16_t>(op, weight_bf16, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
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
        bin_os_->write(weightData, weight_bf16.size() * sizeof(uint16_t));
      }
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

      isRedundant = isRedundantWeight<uint32_t>(op, weight_uint32, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
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
        bin_os_->write(weightData, weight_uint32.size() * sizeof(uint32_t));
      }
    } else if (weightOp.storage() == "FP32") {
      std::vector<float> weight_fp32;
      auto weight = wTF->readTensor<float>(tensor_name, type);
      weight_fp32.assign(weight->begin(), weight->end());
      size = weight_fp32.size() * sizeof(float);

      isRedundant = isRedundantWeight<float>(op, weight_fp32, md5AddrMap_, curPos, md5);
      if (!isRedundant) {
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
        bin_os_->write(weightData, weight_fp32.size() * sizeof(float));
      }
    } else if (weightOp.storage() == "NONE") {
      return success();
    } else {
      llvm::errs() << tensor_name << " weight storage type "
                   << weightOp.storage() << "\n";
      assert(0 && "not supported weight storage type");
    }

    // assign the address to weightOp
    tensor_name += "_";
    tensor_name += md5.substr(0, 5);

    if (!isRedundant) {
      // checking
      auto newPos = bin_os_->tellp();
      std::string s;
      llvm::raw_string_ostream os(s);
      os << tensor_name << "," << llvm::format_hex(curPos, 10)
          << "," << md5 << "," << size << "\n";
      map_os_->write(os.str().c_str(), os.str().size());
      assert(((curPos % alignment_) == 0) && "Expect aligned curPos");
      assert(((newPos % alignment_) == 0) && "Expect aligned newPos");
    } else {
      llvm::errs() << "remove a redundant weight:" << tensor_name
                   << " with md5:" << md5 << "\n";
      curPos = md5AddrMap_.at(md5);
    }

    weightOp->setAttr("name", rewriter.getStringAttr(tensor_name));
    weightOp->setAttr("md5", rewriter.getStringAttr(md5));
    weightOp->setAttr("offset", rewriter.getI64IntegerAttr(curPos + (((uint64_t)1) << 40)));

    return success();
  }


  std::unique_ptr<std::fstream> &bin_os_;
  std::unique_ptr<std::fstream> &map_os_;
  std::map<std::string, uint64_t> &md5AddrMap_;
  size_t alignment_;
};

template<typename OpTy>
struct WeightSetRedundantPattern : public RewritePattern {
  WeightSetRedundantPattern(MLIRContext *context,
      std::unordered_map<int64_t, std::list<Operation *> > &offset_map)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        offset_map_(offset_map) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto weightOp = cast<OpTy>(op);
    //if (!weightOp.offset().hasValue()) {
    //  // error
    //  return failure();
    //}
    int64_t offset = weightOp.offset().getValue();
    if (offset_map_[offset].size() > 1) {
      weightOp->setAttr("is_redundant", rewriter.getBoolAttr(true));
    }
    return success();
  }
  std::unordered_map<int64_t, std::list<Operation *> > &offset_map_;
};

class AssignWeightAddressPass
    : public mlir::PassWrapper<AssignWeightAddressPass, FunctionPass> {
public:
  explicit AssignWeightAddressPass() {}

  static void checkIfFileGood(std::string &fileName,
                              std::unique_ptr<std::fstream> &stream) {
    if (!stream->is_open()) {
      llvm::errs() << "cannot open output file '" + fileName + "\n";
      assert(0);
    }
  }

  static bool loadAddressMapping(std::string &mapFileName,
                                 std::map<std::string, uint64_t> &addrMapping) {
    auto stream = std::make_unique<std::fstream>(mapFileName.c_str(),
                                                 std::fstream::in);
    if (!stream->is_open()) {
      return false;
    }

    char buf[512];
    while (!stream->eof()) {
      memset(buf, 0, sizeof(buf));
      stream->getline(buf, sizeof(buf));
      StringRef str(buf);
      if (str.empty()) {
        continue;
      }
      SmallVector<StringRef, 4> fields;
      str.split(fields, ',', -1, true);
      auto pos = fields[1].str();
      auto md5 = fields[2].str();
      auto addr = std::stol(pos, nullptr, 16);
      addrMapping[md5] = addr;
    }
    return true;
  }

  static void SetRedundant(FuncOp &fn, MLIRContext *context) {
    std::unordered_map<int64_t, std::list<Operation *> > offset_map;
    fn.walk([&] (Operation * op) {
      if (auto castOp = llvm::dyn_cast<tpu::LoadWeightOp>(op)) {
        int64_t offset = castOp.offset().getValue();
        offset_map[offset].emplace_back(op);
      }
    });
    OwningRewritePatternList patterns;
    patterns.insert<WeightSetRedundantPattern<tpu::LoadWeightOp> >(
        context, offset_map);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

  void runOnFunction() override {
    auto fn = getFunction();
    // create a bin file
    assert((clWeightBinFilename != "-") && "no weight bin file specified");
    assert((clWeightMapFilename != "-") && "no weight map file specified");

    auto flags = std::fstream::out;
    std::map<std::string, uint64_t> addrMapping;

    if (clAppendWeight) {
      if (loadAddressMapping(clWeightMapFilename, addrMapping)) {
        flags = flags | std::fstream::app;
      }
    }
    auto weightBinFile = std::make_unique<std::fstream>(
                            clWeightBinFilename.c_str(),
                            flags | std::fstream::binary);
    checkIfFileGood(clWeightBinFilename, weightBinFile);
    auto weightMapFile = std::make_unique<std::fstream>(
                            clWeightMapFilename.c_str(), flags);
    checkIfFileGood(clWeightMapFilename, weightMapFile);

    OwningRewritePatternList patterns;
    auto *context = &getContext();

    // assign address and generate bin file
    patterns.insert<TpuLoadWeightOpPattern<tpu::LoadWeightOp>,
                    TpuLoadWeightOpPattern<tpu::TL_LG_LoadCoeffOp>
    >(context, weightBinFile, weightMapFile,
      addrMapping, clWeightAlignment);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    SetRedundant(fn, context);

    weightBinFile->close();
    weightMapFile->close();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createAssignWeightAddressPass() {
  return std::make_unique<AssignWeightAddressPass>();
}
